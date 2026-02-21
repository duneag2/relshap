SELECT
  e.row_id,

  e.resource      AS RESOURCE,
  e.mgr_id        AS MGR_ID,
  e.role_rollup_1 AS ROLE_ROLLUP_1,
  e.role_rollup_2 AS ROLE_ROLLUP_2,
  e.role_deptname AS ROLE_DEPTNAME,
  e.role_family_desc AS ROLE_FAMILY_DESC,

  r.role_title    AS ROLE_TITLE,
  r.role_family   AS ROLE_FAMILY,
  r.role_code     AS ROLE_CODE
FROM Employee AS e
JOIN Role AS r
  ON e.role_code = r.role_code
ORDER BY e.row_id;
