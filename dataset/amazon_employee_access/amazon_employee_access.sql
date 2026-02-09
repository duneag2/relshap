
SELECT
  a.raw_row_id,

  a.resource      AS RESOURCE,
  a.mgr_id        AS MGR_ID,
  a.role_rollup_1 AS ROLE_ROLLUP_1,
  a.role_rollup_2 AS ROLE_ROLLUP_2,
  a.role_deptname AS ROLE_DEPTNAME,
  a.role_family_desc AS ROLE_FAMILY_DESC,

  r.role_title    AS ROLE_TITLE,
  r.role_family   AS ROLE_FAMILY,
  r.role_code     AS ROLE_CODE
FROM amazon_employee_access a
JOIN Role r
  ON a.role_code = r.role_code
ORDER BY a.raw_row_id;
