SELECT
  d.SERIALNO,
  d.SPORDER,
  d.AGEP,
  e.COW,
  d.SCHL,
  d.MAR,
  e.OCCP,
  b.POBP,
  d.RELP,
  e.WKHP,
  d.SEX,
  d.RAC1P,
  e.PINCP AS PINCP

FROM person_demographic AS d
JOIN person_employment  AS e
  ON d.SERIALNO = e.SERIALNO AND d.SPORDER = e.SPORDER
JOIN person_birth       AS b
  ON d.SERIALNO = b.SERIALNO AND d.SPORDER = b.SPORDER

WHERE e.PINCP IS NOT NULL