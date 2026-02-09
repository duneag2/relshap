SELECT
  d.SERIALNO,
  d.SPORDER,
  d.AGEP,
  d.SCHL,
  d.MAR,
  d.RELP,
  m.DIS,
  d.ESP,
  b.CIT,
  d.MIG,
  d.MIL,
  d.ANC,
  cn.NATIVITY,
  m.DEAR,
  m.DEYE,
  m.DREM,
  d.SEX,
  d.RAC1P,
  e.ESR AS ESR
FROM person_demographic AS d
JOIN person_employment  AS e
  ON d.SERIALNO = e.SERIALNO AND d.SPORDER = e.SPORDER
JOIN person_birth       AS b
  ON d.SERIALNO = b.SERIALNO AND d.SPORDER = b.SPORDER
JOIN person_medical     AS m
  ON d.SERIALNO = m.SERIALNO AND d.SPORDER = m.SPORDER
LEFT JOIN cit_nativity  AS cn
  ON b.CIT = cn.CIT;