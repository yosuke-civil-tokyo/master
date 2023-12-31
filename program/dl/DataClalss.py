# Create dictionaries for data types
pt_data_types = {
    'レコード区分': int,
    '回収分類': int,
    'バッチ番号': int,
    '整理番号：市区町村': int,
    '整理番号：ロット番号': int,
    '整理番号：世帯ＳＱ': int,
    '世帯人数／5歳未満含む': int,
    '世帯人数／5歳未満除く': int,
    '回収個人票数': int,
    '現住所：完全桁数': int,
    '現住所：ゾーンコード': int,
    '現住所：JISコード（5桁）': int,
    '所有車両：自動車': int,
    '所有車両：自転車': int,
    '所有車両：原付・バイク': int,
    '世帯年収': float,
    '個人番号': int,
    '性別': int,
    '年齢': int,
    '世帯主との続柄': int,
    '就業（形態・状況）': int,
    '職業': int,
    '自動車運転免許保有の状況': int,
    '自由に使える自動車の有無': int,
    '外出に関する身体的な困難さ': int,
    '勤務先・通学先・通園先：完全桁数': int,
    '勤務先・通学先・通園先：ゾーンコード': int,
    '勤務先・通学先・通園先：JISコード（5桁）': int,
    '勤務時間固定の有無': int,
    '勤務先の始業時刻：午前・午後': int,
    '勤務先の始業時刻：時': int,
    '勤務先の始業時刻：分': int,
    '調査対象日の在宅勤務の有無': int,
    'トリップの有無': int,
    'トリップ数': int,
    'トリップ番号': int,
    '出発地：区分': int,
    '出発地：完全桁数': int,
    '出発地：ゾーンコード': int,
    '出発地：JISコード（5桁）': int,
    '施設の種類（出発地）': int,
    '出発時刻：午前・午後': int,
    '出発時刻：時': int,
    '出発時刻：分': int,
    '到着地：区分': int,
    '到着地：完全桁数': int,
    '到着地：ゾーンコード': int,
    '到着地：JISコード（5桁）': int,
    '施設の種類（到着地）': int,
    '到着時刻：午前・午後': int,
    '到着時刻：時': int,
    '到着時刻：分': int,
    '目的地での消費額': float,
    '移動の目的': int,
    '同行人数：人数': int,
    '同行人数：小学生以下の有無': int,
    '同行人数：高齢者の有無': int,
    '交通手段㈰': int,
    '交通手段㈪': int,
    '交通手段㈫': int,
    '交通手段㈬': int,
    '交通手段㈭': int,
    '交通手段㈮': int,
    '交通手段㈯': int,
    '交通手段㉀': int,
    '鉄道利用駅：乗車駅': int,
    '鉄道利用駅：降車駅': int,
    '駐輪場所：㈰': int,
    '駐輪場所：㈪': int,
    '自動車利用：運転有無': int,
    '自動車利用：高速道路利用有無': int,
    '自動車利用：駐車場': int,
    '拡大係数': float,
    '世帯類型': int,
    '発目的': int,
    '着目的': int,
    '目的種類：分類１': int,
    '目的種類：分類２': int,
    '目的種類：分類３': int,
    '代表交通手段：分類０': int,
    '代表交通手段：分類１': int,
    '代表交通手段：分類２': int,
    '代表交通手段：分類３': int,
    'トリップ時間（分）': float,
    '滞在時間（分）': int,
    'マストラ乗車：代表交通手段': int,
    'マストラ乗車：駅コード（施設）': int,
    'マストラ乗車：駅地点（ゾーン）': int,
    'マストラ乗車：端末手段': int,
    'マストラ降車：代表交通手段': int,
    'マストラ降車：駅コード（施設）': int,
    'マストラ降車：駅地点（ゾーン）': int,
    'マストラ降車：端末手段': int,
}


walk_data_types = {
    '発ゾーン': int,
    '着ゾーン': int,
    '徒歩所要時間（分）': float,
}

car_data_types = {
    '発ゾーン': int,
    '着ゾーン': int,
    '時間帯': int,
    '時間（分）': int,
    '走行コスト': float,
    '通行料金': float
}
zone_data_types = {
    'ColumnX': int,
    'ColumnY': float,
    'ColumnZ': str,
    # Add more columns and data types as needed
}