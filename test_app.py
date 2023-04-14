import pytest
import os
import syspend
from bs4 import BeautifulSoup
from app import app

HOME_PAGE = os.path.join(os.getcwd(),'templates', "index.html")

def test_index_route():
    response = app.test_client().get('/')
    with open(HOME_PAGE, 'r') as f:
        expected_content = f.read()
    print(response.data.decode('utf-8'))
    assert response.status_code == 200
    assert response.data.decode('utf-8') == expected_content

if __name__ == '__main__':
    with open(HOME_PAGE, 'r') as f:
        expected_content = f.read()
        print(expected_content)
    response = app.test_client().get('/')
    soup = BeautifulSoup(response.data, 'html.parser')
    html_content = str(soup.html)
    print(html_content)
