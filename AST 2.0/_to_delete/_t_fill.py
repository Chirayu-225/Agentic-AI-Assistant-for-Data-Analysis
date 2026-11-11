import pandas as pd, numpy as np, random
from fill_missing import fill_missing, fill_derived
random.seed(1)
n=600
q=[random.randint(1,12) for _ in range(n)]; pr=[round(random.uniform(5,400),2) for _ in range(n)]
df=pd.DataFrame({"qty":q,"price":pr,"amount":[round(a*b,2) for a,b in zip(q,pr)]})
truth=df.copy()
for i in range(0,40,2): df.loc[i,"amount"]=np.nan
for i in range(1,41,4): df.loc[i,"qty"]=np.nan
o,r=fill_missing(df); assert r["total_filled"]==0, r   # median off
o,e=fill_derived(df)
print(e)
m=df["amount"].isna()
assert np.allclose(o.loc[m,"amount"],truth.loc[m,"amount"],atol=0.011)
m=df["qty"].isna()&df["price"].notna()&df["amount"].notna()
assert (o.loc[m,"qty"]==truth.loc[m,"qty"]).all()
# vacuous + sparse group
d=pd.DataFrame({"k":["a"]*5+["b"]*5+["c","c"]*3+["d"]*5+["e"]*5+["f"]*5, "v":["x"]*40+[None]*2})
d.loc[len(d)]=("a",None)
o,r=fill_missing(d); print(r["filled"])
# group with only 2 agreeing rows must not fill
g=pd.DataFrame({"k":list("aabbbcccddd")*6,"v":["p","p",None,"q","q","r","r","r","s","s","s"]*6})
g.loc[0,"v"]=None
print(fill_missing(g)[1]["filled"])
print("OK")
