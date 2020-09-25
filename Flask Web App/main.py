from flask import Flask, send_file

from plotlydash.dashboard import render_dashboard
from plotlydash.navigation import render_navigation
from plotlydash.prediction import render_prediction
from plotlydash.authentification import render_authentification

app = Flask(__name__)

render_dashboard(app)
render_prediction(app)
render_navigation(app)
render_authentification(app)

@app.route('/download')
def download_csv():
    return send_file('../Web App/outputs/test.csv',
                     mimetype='text/csv',
                     attachment_filename='test.csv',
                     as_attachment=True)

if __name__ == '__main__':
    app.run(debug = True)
