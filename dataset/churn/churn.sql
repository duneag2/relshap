SELECT
  c.state,
  c.account_length,
  c.area_code,
  c.phone_number,
  c.international_plan,
  c.voice_mail_plan,
  c.number_vmail_messages,

  d.total_minutes AS total_day_minutes,
  d.total_calls   AS total_day_calls,
  d.total_charge  AS total_day_charge,

  e.total_minutes AS total_eve_minutes,
  e.total_calls   AS total_eve_calls,
  e.total_charge  AS total_eve_charge,

  n.total_minutes AS total_night_minutes,
  n.total_calls   AS total_night_calls,
  n.total_charge  AS total_night_charge,

  i.total_minutes AS total_intl_minutes,
  i.total_calls   AS total_intl_calls,
  i.total_charge  AS total_intl_charge,

  c.number_customer_service_calls
FROM Customer AS c
LEFT JOIN Usage AS d ON d.customer_id = c.customer_id AND d.period = 'DAY'
LEFT JOIN Usage AS e ON e.customer_id = c.customer_id AND e.period = 'EVE'
LEFT JOIN Usage AS n ON n.customer_id = c.customer_id AND n.period = 'NIGHT'
LEFT JOIN Usage AS i ON i.customer_id = c.customer_id AND i.period = 'INTL';