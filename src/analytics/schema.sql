CREATE TABLE spaces (
  space_id VARCHAR(4) PRIMARY KEY,   -- S1 ... S6
  name VARCHAR(60), source VARCHAR(60), capacity INT,
  use_type VARCHAR(20), quiet BOOLEAN, outlets BOOLEAN,
  x FLOAT, y FLOAT                   -- position on the occupancy map
);
CREATE TABLE occupancy (
  space_id VARCHAR(4), ts DATETIME, count_true FLOAT, count_detector FLOAT,
  PRIMARY KEY (space_id, ts)
);
CREATE TABLE forecasts (
  space_id VARCHAR(4), ts DATETIME, model VARCHAR(20),
  yhat FLOAT, yhat_lower FLOAT, yhat_upper FLOAT,
  PRIMARY KEY (space_id, ts, model)
);
CREATE TABLE reservations (
  id INT AUTO_INCREMENT PRIMARY KEY, space_id VARCHAR(4),
  start_ts DATETIME, end_ts DATETIME
);
CREATE TABLE recommendations (
  id INT AUTO_INCREMENT PRIMARY KEY, created_ts DATETIME,
  request_start DATETIME, request_end DATETIME, activity VARCHAR(20),
  top1 VARCHAR(4), top2 VARCHAR(4), top3 VARCHAR(4)
);