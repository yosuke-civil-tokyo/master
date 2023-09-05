# Data Memo
Taking note of   

- data structure
- data loader process

## data structure
-> for now, refer the 'データレイアウト一覧'

## data loader process
### PT data
1. schedule data
    - personal features
    - schedule type (+ num of trips)
    - trip data
2. trip data
    - trip features
    - activity features

schedule data has a list of trip data

### LOS data
add to each trip level data as one object
- making table of LOS data (each row is o/d/t combination)
- add the corresponding row as LOS object to trip data (start from walk & car)

### zone data
add to each trip level data as one object
- making table of zone (each row represents zone)
- add the corresponding row as Zone object to trip data (origin and destination)