SELECT
  o.order_id,
  o.customer_id,

  rv.review_id,
  rv.review_score,

  o.order_status,
  (o.order_delivered_customer_date - o.order_purchase_timestamp) AS deliver_interval,
  (o.order_delivered_customer_date - o.order_estimated_delivery_date) AS late_interval,
  CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END AS is_late,

  c.customer_state,
  c.customer_city,
  c.customer_zip_code_prefix,

  oi_top.product_id AS top_product_id,
  oi_top.seller_id  AS top_seller_id,
  oi_top.price      AS top_item_price,
  oi_top.freight_value AS top_freight_value,

  oi_top.n_items,
  oi_top.n_products,
  oi_top.n_sellers,
  oi_top.sum_item_price,
  oi_top.sum_freight,

  p.product_category_name,
  tr.product_category_name_english,

  pay.n_pay_rows,
  pay.sum_payment_value,
  pay.max_installments,

  s.seller_state,
  s.seller_city,
  s.seller_zip_code_prefix

FROM olist_orders_dataset AS o

JOIN olist_order_customer_dataset AS c
  ON o.customer_id = c.customer_id

JOIN (
  SELECT
    r.order_id,
    r.review_id,
    r.review_score,
    ROW_NUMBER() OVER (
      PARTITION BY r.order_id
      ORDER BY r.review_creation_date DESC, r.review_id
    ) AS rn
  FROM olist_order_reviews_dataset AS r
  WHERE r.order_id IS NOT NULL
  QUALIFY rn = 1
) AS rv
  ON o.order_id = rv.order_id

LEFT JOIN (
  SELECT
    oi.order_id,
    oi.order_item_id,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,

    COUNT(*) OVER (PARTITION BY oi.order_id) AS n_items,
    COUNT(DISTINCT oi.product_id) OVER (PARTITION BY oi.order_id) AS n_products,
    COUNT(DISTINCT oi.seller_id)  OVER (PARTITION BY oi.order_id) AS n_sellers,
    SUM(oi.price)        OVER (PARTITION BY oi.order_id) AS sum_item_price,
    SUM(oi.freight_value) OVER (PARTITION BY oi.order_id) AS sum_freight,

    ROW_NUMBER() OVER (
      PARTITION BY oi.order_id
      ORDER BY oi.price DESC, oi.order_item_id ASC
    ) AS rn
  FROM olist_order_items_dataset AS oi
  WHERE oi.order_id IS NOT NULL
  QUALIFY rn = 1
) AS oi_top
  ON o.order_id = oi_top.order_id

LEFT JOIN olist_products_dataset AS p
  ON oi_top.product_id = p.product_id

LEFT JOIN product_category_name_translation AS tr
  ON p.product_category_name = tr.product_category_name

LEFT JOIN (
  SELECT
    op.order_id,
    COUNT(*) OVER (PARTITION BY op.order_id) AS n_pay_rows,
    SUM(op.payment_value) OVER (PARTITION BY op.order_id) AS sum_payment_value,
    MAX(op.payment_installments) OVER (PARTITION BY op.order_id) AS max_installments,
    ROW_NUMBER() OVER (
      PARTITION BY op.order_id
      ORDER BY op.payment_sequential ASC
    ) AS rn
  FROM olist_order_payments_dataset AS op
  WHERE op.order_id IS NOT NULL
  QUALIFY rn = 1
) AS pay
  ON o.order_id = pay.order_id

LEFT JOIN olist_sellers_dataset AS s
  ON oi_top.seller_id = s.seller_id

WHERE
  o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
;
