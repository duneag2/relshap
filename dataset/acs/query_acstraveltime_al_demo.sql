SELECT
  d.SERIALNO,
  d.SPORDER,
  d.AGEP,
  d.SCHL,
  d.MAR,
  d.SEX,
  m.DIS,
  d.ESP,
  d.MIG,
  d.RELP,
  d.RAC1P,
  h.PUMA,
  h.ST,
  b.CIT,
  e.OCCP,
  e.JWTR,
  e.POWPUMA,
  e.POVPIP,
  e.JWMNP AS JWMNP
FROM person_demographic AS d
JOIN person_employment  AS e
  ON d.SERIALNO = e.SERIALNO AND d.SPORDER = e.SPORDER
JOIN person_birth       AS b
  ON d.SERIALNO = b.SERIALNO AND d.SPORDER = b.SPORDER
JOIN person_medical     AS m
  ON d.SERIALNO = m.SERIALNO AND d.SPORDER = m.SPORDER
JOIN household          AS h
  ON d.SERIALNO = h.SERIALNO;
