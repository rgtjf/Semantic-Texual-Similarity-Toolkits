# coding: utf8
"""
Ref: 1. https://cloud.google.com/translate/v2/pricing
     2. https://console.cloud.google.com/iam-admin/projects?authuser=0&_ga=1.59488365.1511093086.1475026836

- Usage fees:
  - Translation:
    $20 per 1 million characters of text, where the charges are adjusted in proportion to the number of characters actually provided. For example, if you were to translate 500K characters, you would be billed $10.
  - Language Detection:
    $20 per 1 million characters of text, where the charges are adjusted in proportion to the number of characters actually provided.
- Usage limits:
  - Google Translate API has default limits of 2 million characters/day and 100,000 characters per 100 seconds (average 1000 characters/second). You can increase the first limit up to 50 million characters/day in the Cloud Platform Console by following the instructions below.
  - If you need to translate more than 50 million characters/day or 1000 characters/second, contact us.

To enable billing for your project, do the following:
  1. Visit the Billing page.
  2. If you don't have an active billing account, create one by clicking New billing account and following the instructions.

To view or change usage limits for your project, or to request an increase to your quota, do the following:
  1. If you don't already have a billing account for your project, then create one.
  2. Visit the Enabled APIs page of the API library in the Cloud Platform Console, and select an API from the list.
  3. To view and change quota-related settings, select Quotas. To view usage statistics, select Usage.
"""

# Imports the Google Cloud client library
from google.cloud import translate

api_key = 'AIzaSyAjNIYV8HrDLukNXqKxNyc_QquYJQRhbrA'

# Instantiates a client
translator = translate.Client(api_key)