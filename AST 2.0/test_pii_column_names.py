"""
test_pii_column_names.py — personal columns are recognised even when the name has no separators.

The sales sample has CONTACTFIRSTNAME, CONTACTLASTNAME, ADDRESSLINE1 and ADDRESSLINE2. The token rules only split at
separators and camelCase, so these run-together names matched nothing and their values went to the model; only PHONE was
withheld. Run: python test_pii_column_names.py
"""
import pii_redaction as pii

FAIL = []


def check(label, cond, detail=""):
    print(f"[{'OK' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(label)


SALES = ["ORDERNUMBER", "QUANTITYORDERED", "PRICEEACH", "ORDERLINENUMBER", "SALES", "ORDERDATE", "STATUS", "QTR_ID",
         "MONTH_ID", "YEAR_ID", "PRODUCTLINE", "MSRP", "PRODUCTCODE", "CUSTOMERNAME", "PHONE", "ADDRESSLINE1",
         "ADDRESSLINE2", "CITY", "STATE", "POSTALCODE", "COUNTRY", "TERRITORY", "CONTACTLASTNAME", "CONTACTFIRSTNAME",
         "DEALSIZE"]
PERSONAL = {"PHONE", "ADDRESSLINE1", "ADDRESSLINE2", "CONTACTLASTNAME", "CONTACTFIRSTNAME"}


def withheld(names):
    return {n for n in names if pii.classify_column_name(n) == "personal"}


def main():
    check("sales sample: exactly the personal columns are withheld", withheld(SALES) == PERSONAL, str(sorted(withheld(SALES) ^ PERSONAL)))
    lower = [n.lower() for n in SALES]
    check("same result for the cleaner's lower-case names", withheld(lower) == {n.lower() for n in PERSONAL})
    check("a business column and a company-style name are NOT withheld", pii.classify_column_name("customername") != "personal"
          and pii.classify_column_name("productname") != "personal" and pii.classify_column_name("sales") is None)
    for name in ["contactfirstname", "CustomerLastName", "employeefirstname", "FULLNAME", "emailaddress", "ipaddress",
                 "homephonenumber", "address_line_1", "Address Line 2", "shippingaddressline1", "patientsurname"]:
        check(f"personal: {name}", pii.classify_column_name(name) == "personal")
    for name in ["hostname", "filename", "tablename", "productname", "lastupdate", "last_updated", "firstorderdate",
                 "namespace", "addressable", "lineitem", "linenumber", "orderline", "state", "city", "country"]:
        check(f"not personal: {name}", pii.classify_column_name(name) != "personal")
    check("a bare 'name' stays 'review', not withheld", pii.classify_column_name("name") == "review")
    check("the existing separator rules still work", pii.classify_column_name("first_name") == "personal"
          and pii.classify_column_name("Email") == "personal" and pii.classify_column_name("customer_email") == "personal")

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        raise SystemExit(1)
    print("DONE — all checks passed.")


if __name__ == "__main__":
    main()
