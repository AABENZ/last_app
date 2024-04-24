import pymongo
uri = "mongodb+srv://root:<password>@cluster0.1elqk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# database_url = "mongodb://admin:mysecretpassword@example.com:27017/mydatabase?authSource=admin" 
client = pymongo.MongoClient(uri)

db = client['tunspeech']  # Access the database
collection = db['tunspeech']  # Access a collection

