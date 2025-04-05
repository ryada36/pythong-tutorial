import pytest
from unittest.mock import patch, Mock
from day_4.apis import fetch_posts


@pytest.fixture
def mock_requests_get():
    with patch('day_4.apis.requests.get') as mock_get:
        yield mock_get

def test_fetch_posts_success(mock_requests_get):
    # Mock the response from requests.get
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{'id': 1, 'title': 'Test Post'}]
    mock_requests_get.return_value = mock_response

    # Call the function
    result = fetch_posts()

    # Assert the result
    assert result == [{'id': 1, 'title': 'Test Post'}]
    mock_requests_get.assert_called_once_with('https://jsonplaceholder.typicode.com/posts/1')

        