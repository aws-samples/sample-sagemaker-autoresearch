# Test bed for the SageMaker Auto Research plugin

Everything needed to take an empty AWS account to a working end-to-end test run:
a synthetic multi-table churn dataset in Redshift, the AWS resources the plugin
expects, and a validated `research.yaml`.

This is a demo project and should not be used for production code without additional quality and security reviews.

## What gets created

| Resource | Name | Notes |
|---|---|---|
| S3 bucket | `sm-auto-research-<account>-<region>` | slices, training output, model artifacts. `RemovalPolicy.RETAIN`. |
| IAM role | `SageMakerAutoResearchRole` | trusted by **both** SageMaker and Redshift — see below |
| Redshift Serverless namespace | `autoresearch` | database `autoresearch`, admin password in a managed secret |
| Redshift Serverless workgroup | `autoresearch-wg` | 8 RPU (the minimum), private |

**Reused, not created:** an existing MLflow tracking server, if the account has
one (discovered at synth time). The role is granted access to it and to its
artifact-store bucket. This avoids standing up a second billable server. See
step 2 for how to pin or disable that.

### Why one role for two services

`scripts/sm_research/redshift_unload.py` passes `sagemaker.role_arn` directly into
`UNLOAD ... IAM_ROLE '<arn>'`. So the SageMaker execution role and the Redshift
UNLOAD role must be the *same* role. A typical Redshift role trusts only
`redshift.amazonaws.com` and so cannot serve as a SageMaker execution role, which
is why the stack creates a new dual-trust role instead of reusing one.

## The dataset

Synthetic churn data, 40,000 customers. Signal is deliberately **split across
tables** through interaction terms, so a model built on any one table plateaus
below the goal while the joined data clears it.

```
churn.customers      customer_id, age, region, signup_channel, tenure_months     40,000
churn.subscriptions  customer_id, plan_tier, monthly_price, contract_type, autopay 40,000
churn.usage_monthly  customer_id, month_num, avg_minutes, support_tickets,
                     late_payments, data_gb                                      480,000  (12 mo/customer)
churn.labels         customer_id, churned                                         40,000
```

Churn base rate ≈ 0.359. Three cross-table interactions drive the label:
month-to-month **and** heavy support load; short tenure **and** no autopay; long
tenure **and** a clean payment record.

`churn.usage_monthly` is a monthly panel, so a usable feature set requires a
`GROUP BY customer_id` aggregation — not just a join.

### Measured F1 ceilings

Reference models (`--report`), held-out split, tuned decision threshold:

| Feature set | F1 |
|---|---|
| subscriptions only | 0.539 |
| usage only (aggregated) | 0.601 |
| customers only | 0.615 |
| customers + subscriptions | 0.618 |
| **all three joined** | **0.730** |

`research.yaml` sets **`threshold: 0.70`** — above every single-table ceiling,
below the joined ceiling. The loop cannot hit the goal by luck; it has to find
the join and the aggregation.

## Runbook

### 1. Generate the CSVs

```bash
cd testdata
uv run generate_churn_data.py --report
```

Writes `testdata/csv/*.csv` and prints the ceilings above. Deterministic
(`--seed 42`). Re-run with `--report` any time you change the generator — if the
ceilings move, update `THRESHOLD` in `write_research_yaml.py`.

### 2. Deploy the stack

```bash
cd infra
export AWS_PROFILE=oregon
export CDK_DEFAULT_REGION=us-west-2
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
npx --yes aws-cdk@2.1133.0 deploy
```

Nothing account-specific is hardcoded. At synth time the stack discovers a
default-VPC subnet in each of three AZs (Redshift Serverless requires three),
and reuses an existing MLflow tracking server if the account has one. Override
any of that with context flags:

```bash
-c subnet_ids=subnet-a,subnet-b,subnet-c   # skip subnet discovery
-c mlflow_arn=arn:aws:sagemaker:...        # use a specific tracking server
-c mlflow_bucket=amzn-s3-demo-artifact-bucket        # its artifact store bucket
-c mlflow_arn=none                         # skip MLflow reuse entirely
```

With `mlflow_arn=none` (or in an account with no server) the stack omits the
MLflow grants and leaves `tracking_server_arn` blank in `research.yaml`, so the
plugin will ask before creating one — a billable resource.

The pinned CLI version is required: an older global `cdk` (e.g. 2.1031.2) cannot
read the cloud assembly schema (54.0.0) that this `aws-cdk-lib` emits.

If the account has never been bootstrapped for CDK, run `bootstrap` first with
the same env vars.

### 3. Load the data into Redshift

Read the values from the deployed stack rather than pasting them:

```bash
cd testdata
export AWS_PROFILE=oregon
export REGION=us-west-2

out() { aws cloudformation describe-stacks --stack-name AutoResearchTestBed \
  --region "$REGION" --query \
  "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }

SECRET=$(aws redshift-serverless get-namespace --namespace-name autoresearch \
  --region "$REGION" --query 'namespace.adminPasswordSecretArn' --output text)

uv run --with boto3 load_to_redshift.py \
  --profile "$AWS_PROFILE" --region "$REGION" \
  --bucket "$(out S3Bucket)" \
  --workgroup "$(out WorkgroupName)" --database "$(out DatabaseName)" \
  --secret-arn "$SECRET" \
  --role-arn "$(out RoleArn)"
```

Uploads the CSVs, creates schema `churn`, then `COPY`s each table and prints row
counts. Idempotent — it drops and recreates the tables, so re-running is safe.

**A cold serverless workgroup takes several minutes to resume.** The first
statement in this script measured ~4.5 minutes on the pre-existing workgroup.
The script waits up to 15 minutes per statement.

### 4. Write research.yaml

```bash
uv run --with boto3 --with pyyaml write_research_yaml.py \
  --profile oregon --region us-west-2 --out ../research.yaml
```

Reads the stack outputs, discovers the admin secret, writes the config, and
validates it with the plugin's own `sm_research.config.validate`.

### 5. Test the plugin

```bash
/sm-research --dry-run     # cheap single-experiment validation first
/sm-research               # full loop
```

`research.yaml` already exists, so the interview should short-circuit. Budget is
capped at `max_total_cost_usd: 25`, `max_wallclock_hours: 3`, 4 rounds × 3
experiments.

## Cost

- Redshift Serverless 8 RPU: billed per second while queries run, 60s minimum
  charge per query. Idle costs nothing (compute); storage is a few cents/month
  for this data.
- SageMaker training: `ml.m5.xlarge` ≈ $0.23/hr. Jobs on this data run minutes.
- MLflow tracking server: **already running and billing** in this account
  (~$0.50–1.00/hr for `Small`) regardless of this test.
- The loop's own `max_total_cost_usd: 25` cap only counts SageMaker training
  time; it does not see Redshift or MLflow charges.

## Teardown

The stack does not delete anything on its own — the bucket is `RETAIN`.

To stop Redshift charges, delete the workgroup and namespace **manually and
deliberately** (this destroys the test data; the S3 bucket and CSVs survive, so
step 3 can rebuild it):

```bash
aws redshift-serverless delete-workgroup --workgroup-name autoresearch-wg \
  --profile oregon --region us-west-2
# wait for the workgroup to finish deleting, then:
aws redshift-serverless delete-namespace --namespace-name autoresearch \
  --final-snapshot-name '' --profile oregon --region us-west-2
```

Any pre-existing Redshift workgroups and the reused MLflow tracking server are
untouched by any of this — the stack only ever deletes what it created, and the
artifact bucket is retained even then.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
