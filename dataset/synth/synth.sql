SELECT
  o.order_id,
  o.customer_id,
  c.age,
  c.gender,
  c.region,
  c.currency,
  c.life_stage,
  o.channel,

  oi.tier AS top_tier,
  oi.unit_price AS top_unit_price,

  COUNT(*) OVER (PARTITION BY o.order_id) AS n_items,
  SUM(oi.qty) OVER (PARTITION BY o.order_id) AS total_qty,

  o.order_total,
FROM Orders AS o
JOIN Customer AS c
  ON o.customer_id = c.customer_id
JOIN Item AS oi
  ON o.order_id = oi.order_id
WHERE o.order_total >= 0
QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY o.order_id
    ORDER BY oi.unit_price DESC, oi.line_no ASC
  ) = 1
;