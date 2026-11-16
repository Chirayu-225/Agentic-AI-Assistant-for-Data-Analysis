# Evaluation gate

| Suite | Asked | Correct | Verified + correct | Verified + WRONG | Unverified correct | Unverified wrong | Abstained | Failed | Gate |
|---|---|---|---|---|---|---|---|---|---|
| sample_sales | 9 | 9 | 5 | 0 | 4 | 0 | 0 | 0 | pass |
| expenses | 14 | 14 | 12 | 0 | 2 | 0 | 0 | 0 | pass |
| invoices | 19 | 19 | 15 | 0 | 4 | 0 | 0 | 0 | pass |
| store_units | 14 | 14 | 4 | 0 | 10 | 0 | 0 | 0 | pass |

## sample_sales

- Gate passed.
- By route: generated_python 1/1, query_plan 5/5, shortcut 3/3

| ID | Result | Trust | Route | Question | Expected | Got / reason |
|---|---|---|---|---|---|---|
| Q1 | ok | unverified_correct | shortcut | How many rows are in the dataset? | 24 | 24 |
| Q2 | ok | unverified_correct | shortcut | How many columns are there? | 6 | 6 |
| Q3 | ok | unverified_correct | shortcut | What is the total revenue? | 74100 | 74100 |
| Q4 | ok | verified_correct | query_plan | What is the average units per order? | 5.25 | 5.25 |
| Q5 | ok | verified_correct | query_plan | Which region has the highest total revenue? | South | table[1 rows] [{"region": "South", "total_revenue": 22800}] |
| Q6 | ok | verified_correct | query_plan | List the regions ranked by total revenue | South/East/North/West | table[4 rows] [{"region": "South", "total": 22800}, {"region": "East", "total": 19200}] |
| Q7 | ok | verified_correct | query_plan | How many distinct products are there? | 3 | 3 |
| Q8 | ok | verified_correct | query_plan | How many orders were for Phone? | 8 | 8 |
| Q9 | ok | unverified_correct | generated_python | Plot total revenue by region as a bar chart |  | [chart] |

## expenses

- Gate passed.
- By route: generated_python 2/2, query_plan 12/12

| ID | Result | Trust | Route | Question | Expected | Got / reason |
|---|---|---|---|---|---|---|
| EXP01 | ok | verified_correct | query_plan | How many unique expense claims are there? | 300 | 300 |
| EXP02 | ok | verified_correct | query_plan | What is the net total of all expenses including refunds? | 1201775.11 | 1.20178e+06 |
| EXP03 | ok | verified_correct | query_plan | How many refunds (negative amounts) are there? | 26 | 26 |
| EXP04 | ok | verified_correct | query_plan | Which department has the highest net expense total? | Sales | table[1 rows] [{"department": "Sales", "total": 407239.9000000001}] |
| EXP05 | ok | verified_correct | query_plan | What is the average expense amount, excluding refunds? | 4508.73 | 4508.73 |
| EXP06 | ok | verified_correct | query_plan | Which expense category has the largest net total? | Travel | table[1 rows] [{"category": "Travel", "total": 409739.1499999999}] |
| EXP07 | ok | verified_correct | query_plan | How many distinct employees submitted expenses? | 60 | 60 |
| EXP08 | ok | verified_correct | query_plan | What is the largest single expense? | 15871.31 | 15871.3 |
| EXP09 | ok | verified_correct | query_plan | What is the net Travel spend in the Sales department? | 128141.96 | 128142 |
| EXP10 | ok | verified_correct | query_plan | How many expenses were submitted in May 2026? | 73 | 73 |
| EXP11 | ok | unverified_correct | generated_python | What percentage of net spend is Software? | 18.1 | 18.0617 |
| EXP12 | ok | verified_correct | query_plan | List departments ranked by net expense total, highest first | Sales/Engineering/Operations/Finance | table[4 rows] [{"department": "Sales", "net_expense": 407239.9000000001}, {"department": " |
| EXP13 | ok | verified_correct | query_plan | Show all refunds | 26 | 26 rows |
| EXP14 | ok | unverified_correct | generated_python | Plot net expense total by department as a bar chart |  | [chart] |

## invoices

- Gate passed.
- By route: generated_python 3/3, query_plan 15/15, shortcut 1/1

| ID | Result | Trust | Route | Question | Expected | Got / reason |
|---|---|---|---|---|---|---|
| INV01 | ok | verified_correct | query_plan | How many invoices are there? | 400 | 400 |
| INV02 | ok | unverified_correct | shortcut | What is the total invoiced amount? | 13748034.32 | 1.3748e+07 |
| INV03 | ok | verified_correct | query_plan | What is the average invoice amount? | 34370.09 | 34370.1 |
| INV04 | ok | verified_correct | query_plan | Which region has the highest total invoiced amount? | North | table[1 rows] [{"region": "North", "total_amount": 4853917.909999996}] |
| INV05 | ok | verified_correct | query_plan | List regions ranked by total invoiced amount, highest first | North/South/East/West | table[4 rows] [{"region": "North", "total_amount": 4853917.909999996}, {"region": "South", |
| INV06 | ok | verified_correct | query_plan | How many distinct regions are there? | 4 | 4 |
| INV07 | ok | verified_correct | query_plan | How many invoices are overdue? | 65 | 65 |
| INV08 | ok | verified_correct | query_plan | What is the total amount of overdue invoices? | 2292162.28 | 2.29216e+06 |
| INV09 | ok | verified_correct | query_plan | Which customer segment has the highest average invoice amount? | Enterprise | table[1 rows] [{"segment": "Enterprise", "avg_amount": 86990.49878048782}] |
| INV10 | ok | unverified_correct | generated_python | What percentage of total invoiced amount comes from the Enterprise seg | 51.9 | 51.8854 |
| INV11 | ok | verified_correct | query_plan | What is the median days to pay for paid invoices? | 42.0 | 42 |
| INV12 | ok | verified_correct | query_plan | How many invoices have no days_to_pay value? | 113 | 113 |
| INV13 | ok | unverified_correct | generated_python | Which month had the highest total invoiced amount? Answer as YYYY-MM. | 2026-06 | 2026-06 |
| INV14 | ok | verified_correct | query_plan | What was the total invoiced amount in March 2026? | 1739978.04 | 1.73998e+06 |
| INV15 | ok | verified_correct | query_plan | How many invoices are larger than 50,000? | 86 | 86 |
| INV16 | ok | verified_correct | query_plan | What is the average discount percentage? | 3.91 | 3.9125 |
| INV17 | ok | verified_correct | query_plan | How many pending invoices are in the South region? | 19 | 19 |
| INV18 | ok | verified_correct | query_plan | Show all overdue invoices in the North region | 21 | 21 rows |
| INV19 | ok | unverified_correct | generated_python | Plot total invoiced amount by month as a line chart |  | [chart] |

## store_units

- Gate passed.
- By route: generated_python 10/10, query_plan 4/4

| ID | Result | Trust | Route | Question | Expected | Got / reason |
|---|---|---|---|---|---|---|
| STO01 | ok | verified_correct | query_plan | How many stores are there? | 36 | 36 |
| STO02 | ok | verified_correct | query_plan | What were total units sold in March across all stores? | 21098 | 21098 |
| STO03 | ok | unverified_correct | generated_python | Which store sold the most units over the whole year? | ST112 | ST112 |
| STO04 | ok | verified_correct | query_plan | Which city has the most stores? | Mumbai | table[1 rows] [{"city": "Mumbai", "store_count": 11}] |
| STO05 | ok | unverified_correct | generated_python | Which month had the highest total units sold across all stores? | Nov | nov |
| STO06 | ok | unverified_correct | generated_python | What is the average monthly units sold per store, across all stores an | 621.68 | 621.683 |
| STO07 | ok | unverified_correct | generated_python | How many store-months had zero units sold? | 16 | 16 |
| STO08 | ok | unverified_correct | generated_python | What were total annual units for stores in Bengaluru? | 61190 | 61190 |
| STO09 | ok | unverified_correct | generated_python | By what percentage did total units in December exceed total units in J | 71.9 | 71.9479 |
| STO10 | ok | unverified_correct | generated_python | How many stores sold more than 10,000 units in the year? | 7 | 7 |
| STO11 | ok | unverified_correct | generated_python | List the top 3 cities by total annual units | Mumbai/Bengaluru/Delhi | table[3 rows] [{"city": "Mumbai", "annual_total": 86915}, {"city": "Bengaluru", "annual_to |
| STO12 | ok | unverified_correct | generated_python | What share of the year's units were sold in Q4 (Oct-Dec)? Give a perce | 32.1 | 32.0784 |
| STO13 | ok | verified_correct | query_plan | Show the stores in Pune | 4 | 4 rows |
| STO14 | ok | unverified_correct | generated_python | Plot total units by month as a bar chart |  | [chart] |

Verified + WRONG is the number that matters: answers the system told the user it had cross-checked, which were wrong. The cross-check proves the plan was computed correctly, not that the plan matches the question.
