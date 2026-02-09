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
FROM orders AS o
JOIN customer AS c
  ON o.customer_id = c.customer_id
JOIN order_item AS oi
  ON o.order_id = oi.order_id
WHERE o.order_total >= 0
QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY o.order_id
    ORDER BY oi.unit_price DESC, oi.line_no ASC
  ) = 1
;

  -- (o.order_total + 0) AS order_total_copy,
  -- ABS(o.order_total - oi.unit_price) AS total_minus_topprice_abs
-- SELECT
--   o.order_id,
--   o.customer_id,
--   c.age,
--   c.gender,
--   c.region,
--   c.currency,
--   c.life_stage,
--   o.channel,

--   oi.tier AS top_tier,
--   oi.unit_price AS top_unit_price,

--   COUNT(*) OVER (PARTITION BY o.order_id) AS n_items,
--   SUM(oi.qty) OVER (PARTITION BY o.order_id) AS total_qty,

--   o.order_total
-- FROM orders AS o
-- JOIN customer AS c
--   ON o.customer_id = c.customer_id
-- JOIN order_item AS oi
--   ON o.order_id = oi.order_id
-- WHERE
--   o.order_total >= 0

--   AND ((c.life_stage != 'adult') OR (c.age BETWEEN 18 AND 64))
--   AND ((c.life_stage != 'child') OR (c.age BETWEEN 0  AND 17))
--   AND ((c.life_stage != 'senior') OR (c.age >= 65))

--   AND ((c.region != 'EU') OR (c.currency IN ('EUR')))
--   AND ((c.region != 'ASIA') OR (c.currency IN ('TWD')))
--   AND ((c.region != 'NAM') OR (c.currency IN ('CAD')))

-- QUALIFY
--   ROW_NUMBER() OVER (
--     PARTITION BY o.order_id
--     ORDER BY oi.unit_price DESC, oi.line_no ASC
--   ) = 1
-- ;


-- SELECT
--   o.order_id,
--   o.customer_id,
--   c.age,
--   c.gender,
--   c.region,
--   c.currency,
--   c.life_stage,
--   o.channel,
--   oi.tier AS top_tier,
--   oi.unit_price AS top_unit_price,
--   COUNT(*) OVER (PARTITION BY o.order_id) AS n_items,
--   SUM(oi.qty) OVER (PARTITION BY o.order_id) AS total_qty,
--   o.order_total
-- FROM orders AS o
-- JOIN customer AS c
--   ON o.customer_id = c.customer_id
--  AND ((c.region != 'EU') OR (c.currency IN ('EUR')))
-- JOIN order_item AS oi
--   ON o.order_id = oi.order_id
-- WHERE
--   o.order_total >= 0
--   AND CASE
--         WHEN c.life_stage = 'adult' THEN (c.age BETWEEN 18 AND 64)
--         ELSE TRUE
--       END
-- QUALIFY
--   ROW_NUMBER() OVER (
--     PARTITION BY o.order_id
--     ORDER BY oi.unit_price DESC, oi.line_no ASC
--   ) = 1
-- ;
