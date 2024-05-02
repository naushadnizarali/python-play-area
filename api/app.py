from flask import Flask
import logging

app = Flask(__name__)

console_handler = logging.StreamHandler()
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# console_handler.setFormatter(formatter)
# logging.basicConfig(filename="record.log", level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)
# app.logger.addHandler(console_handler)


@app.route("/")
def main():
    # showing different logging levels
    app.logger.error("error message")
    app.logger.info("error message")
    return "testing logging levels."


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5001
    app.logger.info(f"Running on {host}:{port}")
    app.run(debug=True, host=host, port=port)
