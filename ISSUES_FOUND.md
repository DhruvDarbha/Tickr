# Code Review: Issues Found and Fixes Needed

## Critical Issues

### 1. **fetch_historical_prices() - Column Selection Failure**
**Line 100**: If yfinance returns different column names or missing columns, the column selection will fail with KeyError.
**Fix**: Add validation to check columns exist before selection.

### 2. **fetch_historical_prices() - Timezone Assumption**
**Line 91**: Assumes US/Eastern timezone for all stocks. Will fail for non-US stocks or if timestamp already has timezone.
**Fix**: Better timezone detection or handle more gracefully.

### 3. **fetch_latest_price() - None Info Dictionary**
**Line 142**: `stock.info` can return None or empty dict, causing AttributeError on `.get()`.
**Fix**: Add None check before accessing info.

### 4. **fetch_company_news() - JSON Parsing Failure**
**Line 298**: `response.json()` can fail if response is not valid JSON (e.g., HTML error page).
**Fix**: Add try-except around JSON parsing and check Content-Type.

### 5. **fetch_company_news() - Empty Available Dates**
**Line 247-249**: If `available_dates` is empty, `random.sample()` will raise ValueError.
**Fix**: Check if list is empty before calling random.sample().

### 6. **fetch_company_news() - No API Key Validation**
**Line 190**: No validation that marketaux_api_key is not None or empty.
**Fix**: Add input validation at function start.

### 7. **fetch_company_news() - Rate Limiting**
**Line 287**: Making many API calls in loop could hit rate limits. No retry logic or delays.
**Fix**: Add rate limiting, retry logic, or delays between requests.

### 8. **fetch_quarterly_financials() - Empty Columns**
**Line 447**: If `financials.columns` is empty, loop won't run but no error raised.
**Fix**: Check if columns exist before iterating.

### 9. **ingest_all_data() - No Error Handling**
**Lines 565-568**: If any function fails, entire function fails. No partial results.
**Fix**: Add try-except for each function call, return partial results.

### 10. **ingest_all_data() - No Input Validation**
**Line 542**: No validation for empty/None ticker or API key.
**Fix**: Add input validation.

## Medium Priority Issues

### 11. **Date Range Validation**
**fetch_historical_prices()**: No check that start_date < end_date.
**Fix**: Add validation.

### 12. **Future Dates in Specific Dates**
**fetch_company_news()**: Specific dates include 2025 dates which may not have data yet.
**Fix**: Filter out future dates that are beyond end_date.

### 13. **Empty DataFrame After Cleaning**
**All functions**: After dropping NaN/duplicates, DataFrame could become empty. Should handle gracefully.
**Fix**: Check if empty and return appropriate empty DataFrame with correct columns.

### 14. **Network Timeout Handling**
**fetch_company_news()**: 30 second timeout may not be enough for slow connections.
**Fix**: Consider configurable timeout or retry with backoff.

## Low Priority Issues

### 15. **Ticker Format Validation**
**All functions**: No validation that ticker is valid format (e.g., not empty, reasonable length).
**Fix**: Add basic validation.

### 16. **Missing Column Handling**
**fetch_historical_prices()**: If yfinance changes column names, rename will fail silently.
**Fix**: More robust column name handling.

### 17. **No Logging**
**All functions**: No logging for debugging or monitoring.
**Fix**: Add logging for important operations.

