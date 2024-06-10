import logging
from resources import app

logging.basicConfig(level=logging.INFO)
if __name__ == '__main__':
    app.run(debug=True, port=8100)
