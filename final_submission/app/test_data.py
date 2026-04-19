import sqlite3

conn = sqlite3.connect('data/listings.db')
print(conn.execute('SELECT COUNT(*) FROM listings WHERE feature_furnished=1').fetchone())
print(conn.execute('SELECT COUNT(*) FROM listings WHERE feature_garden=1').fetchone())