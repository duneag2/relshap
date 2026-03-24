SELECT
  p.p_id,
  p.inPhase AS inPhase,
  p.yearsInProgram,

  COUNT(DISTINCT ab.p_id_dummy) AS n_advisors,

  COUNT(DISTINCT tb.course_id) AS n_courses,
  COUNT(DISTINCT CASE WHEN c.courseLevel = 'Level_100' THEN tb.course_id END) AS n_courses_level_100,
  COUNT(DISTINCT CASE WHEN c.courseLevel = 'Level_300' THEN tb.course_id END) AS n_courses_level_300,
  COUNT(DISTINCT CASE WHEN c.courseLevel = 'Level_400' THEN tb.course_id END) AS n_courses_level_400,
  COUNT(DISTINCT CASE WHEN c.courseLevel = 'Level_500' THEN tb.course_id END) AS n_courses_level_500

FROM person AS p
LEFT JOIN advisedBy AS ab
  ON ab.p_id = p.p_id
LEFT JOIN person AS a
  ON a.p_id = ab.p_id_dummy
LEFT JOIN taughtBy AS tb
  ON tb.p_id = p.p_id
LEFT JOIN course AS c
  ON c.course_id = tb.course_id

WHERE
  p.student = '1'
  AND p.inPhase IS NOT NULL

GROUP BY
  p.p_id,
  p.student,
  p.inPhase,
  p.yearsInProgram

ORDER BY
  p.p_id
;