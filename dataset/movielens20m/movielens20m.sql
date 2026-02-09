SELECT
  r.userId,
  r.movieId,
  r.rating,
  m.genres,
  t.top_tagId,
  t.top_relevance
FROM rating AS r
JOIN movie AS m
  ON r.movieId = m.movieId
LEFT JOIN (
  SELECT
    gs.movieId        AS movieId,
    gs.tagId          AS top_tagId,
    gs.relevance      AS top_relevance
  FROM genome_scores AS gs
  QUALIFY
    ROW_NUMBER() OVER (
      PARTITION BY gs.movieId
      ORDER BY gs.relevance DESC, gs.tagId ASC
    ) = 1
) AS t
  ON r.movieId = t.movieId
;
