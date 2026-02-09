SELECT
  s.s_suppkey AS suppkey,
  s.s_acctbal AS supp_acctbal,
  n.n_name AS supp_nation,
  r.r_name AS supp_region,

  COUNT(l.l_orderkey) AS n_lineitems,
  COUNT(DISTINCT l.l_orderkey) AS n_orders,
  COUNT(DISTINCT o.o_custkey) AS n_customers,
  COUNT(DISTINCT l.l_partkey) AS n_parts,

  SUM(l.l_extendedprice) AS gross_extendedprice,
  SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue_net,
  AVG(l.l_discount) AS avg_discount,
  AVG(l.l_tax) AS avg_tax,
  AVG(l.l_quantity) AS avg_quantity,

  AVG(ps.ps_supplycost) AS avg_supplycost,
  SUM(l.l_quantity * ps.ps_supplycost) AS est_cost,
  (SUM(l.l_extendedprice * (1 - l.l_discount)) - SUM(l.l_quantity * ps.ps_supplycost)) AS est_margin,

  AVG(date_diff('day', l.l_commitdate, l.l_receiptdate)) AS avg_delivery_delay_days,
  AVG(date_diff('day', l.l_shipdate,   l.l_commitdate)) AS avg_ship_to_commit_days,

  AVG(CASE WHEN l.l_receiptdate > l.l_commitdate THEN 1.0 ELSE 0.0 END) AS late_rate,
  AVG(CASE WHEN l.l_returnflag = 'R' THEN 1.0 ELSE 0.0 END) AS return_rate,

  AVG(CASE WHEN o.o_orderpriority = '1-URGENT' THEN 1.0 ELSE 0.0 END) AS frac_urgent,
  AVG(CASE WHEN o.o_orderpriority = '2-HIGH' THEN 1.0 ELSE 0.0 END) AS frac_high,
  AVG(CASE WHEN o.o_orderpriority = '3-MEDIUM' THEN 1.0 ELSE 0.0 END) AS frac_medium,
  AVG(CASE WHEN o.o_orderpriority = '4-NOT SPECIFIED' THEN 1.0 ELSE 0.0 END) AS frac_not_specified,
  AVG(CASE WHEN o.o_orderpriority = '5-LOW' THEN 1.0 ELSE 0.0 END) AS frac_low

FROM supplier s
JOIN nation  n ON n.n_nationkey = s.s_nationkey
JOIN region  r ON r.r_regionkey = n.n_regionkey

JOIN lineitem l ON l.l_suppkey  = s.s_suppkey
JOIN orders   o ON o.o_orderkey = l.l_orderkey

JOIN partsupp ps
  ON ps.ps_partkey = l.l_partkey
 AND ps.ps_suppkey = l.l_suppkey

GROUP BY
  s.s_suppkey, s.s_acctbal, n.n_name, r.r_name
ORDER BY
  s.s_suppkey;
