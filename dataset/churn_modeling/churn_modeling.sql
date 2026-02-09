SELECT
  d.RowNumber,
  c.CustomerId,
  c.Surname,
  d.CreditScore,
  c.Geography,
  c.Gender,
  c.Age,
  d.Tenure,
  d.Balance,
  d.NumOfProducts,
  d.HasCrCard,
  d.IsActiveMember,
  d.EstimatedSalary
FROM Profile AS d
JOIN Customer AS c
  ON d.CustomerId = c.CustomerId;
