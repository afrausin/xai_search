# xAI Search Website

A modern, user-friendly web interface for the xAI Search API that allows you to search across Web, News, and X (Twitter) with customizable date ranges.

## Features

- 🔍 **Multi-Source Search**: Search across Web, News, and X (Twitter) platforms
- 🌍 **Country Filtering**: Filter Web and News results by country/region (ISO alpha-2)
- 📅 **Date Range Filtering**: Specify start and end dates for your searches
- 📎 **Source Citations**: View clickable URLs of all sources used in the AI summary
- 🎯 **Custom System Prompts**: Control how Grok responds to your searches
- 📊 **Batch Monthly Search**: Run sequential searches for multiple months with rolling date windows
- 💾 **CSV Export**: Save batch search results to CSV files for analysis
- 🎨 **Modern UI**: Beautiful, clean minimalistic design
- 🔒 **Secure**: API key stored server-side in .env file (never exposed to browser)
- ⚡ **Fast**: Real-time search results powered by Grok

## Getting Started

### Prerequisites

1. An xAI API key from [xAI Console](https://console.x.ai/)
2. Node.js installed on your system (v16 or higher)

### Installation

1. Clone or download this repository

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   ```

4. Add your xAI API key to the `.env` file:
   ```
   XAI_API_KEY=your_actual_api_key_here
   PORT=3000
   ```

5. Start the server:
   ```bash
   npm start
   ```

   For development with auto-reload:
   ```bash
   npm run dev
   ```

6. Open your browser and navigate to `http://localhost:3000`

⚠️ **Important**: Always access the app via `http://localhost:3000`, not by opening `index.html` directly. The app requires the backend server to function.

### Usage

1. **Customize system prompt** (optional): Modify how Grok responds to your searches:
   - Default: "You are a helpful search assistant. Provide clear, concise summaries of search results."
   - Examples:
     - "Provide bullet-point summaries only"
     - "Act as a market research analyst and provide detailed insights"
     - "Summarize in a conversational, easy-to-understand tone"

2. **Enter your search query**: Type what you want to search for in the search query field.

3. **Select search sources**: Choose one or more sources:
   - **Web**: General web search
   - **News**: News articles
   - **X**: Posts from X (formerly Twitter)

4. **Select country/region** (optional): Filter Web and News results by country:
   - Uses ISO alpha-2 country codes (US, GB, CA, AU, etc.)
   - Only affects Web and News sources (X searches globally)

5. **Set date range** (optional): Select start and end dates to filter results by time period.
   - ⚠️ **Known Issue**: Date filtering works reliably for X (Twitter) but may be inconsistent for News sources (xAI API limitation)

6. **Adjust max results**: Set the maximum number of results to retrieve (1-29, xAI API limitation).

7. **Click Search**: View your results with AI-generated summaries based on your custom prompt and source citations.

### Batch Monthly Search

For analyzing trends over time, use the **Batch Monthly Search** feature:

1. **Access**: Click "Batch Monthly Search →" button on the main page
2. **Set query template**: Use `{month-year}` placeholder (e.g., "Did Wayfair run promotions in {month-year}?")
3. **Select source**: Choose Web, News, or X (Twitter) - **Recommended: X for reliable date filtering**
4. **Set date range**: Choose start and end months
5. **Configure lookback period**: Set how many months to look back (default: 3, range: 1-12)
   - Example: For March 2024 with 3-month lookback → searches Dec 1, 2023 to Mar 31, 2024
   - Example: For March 2024 with 6-month lookback → searches Sep 1, 2023 to Mar 31, 2024
6. **Run batch search**: Processes each month sequentially with a progress bar
7. **Results**: Downloads as CSV and shows in "Previous Searches" list

**CSV Output includes:**
- Month
- Query used
- Date range searched
- Sources found
- AI answer
- Citations

## Architecture

This application uses a Node.js/Express backend to securely handle API requests:

- **Frontend**: Pure HTML/CSS/JavaScript (no framework required)
- **Backend**: Node.js with Express
- **API Proxy**: Backend proxies requests to xAI API
- **Security**: API key stored in `.env` file (never exposed to browser)

### Search Parameters

The app sends the following search parameters:

```javascript
{
  "mode": "on",           // Enable search
  "sources": [            // Array of selected sources
    { "type": "web" },    // Web search
    { "type": "news" },   // News search
    { "type": "x" }       // X (Twitter) search
  ],
  "from_date": "YYYY-MM-DD",     // Start date (ISO 8601)
  "to_date": "YYYY-MM-DD",       // End date (ISO 8601)
  "max_search_results": 10       // Max results to return
}
```

## Security Notes

✅ **Secure by design**: 
- API key is stored in `.env` file on the server
- Never exposed to the browser or client-side code
- Backend acts as a secure proxy to xAI API

⚠️ **Additional security for production**:
- Implement rate limiting
- Add user authentication
- Use HTTPS
- Set up CORS restrictions
- Add request validation and sanitization

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Features in Detail

### Multi-Source Search
Search across different platforms simultaneously by selecting multiple sources. Each source provides unique results:
- **Web**: General internet search
- **News**: Latest news articles
- **X**: Real-time posts from X platform

### Date Filtering
Filter results by specifying a date range. Perfect for:
- Historical research
- Tracking trends over time
- Finding specific events or announcements

### AI-Powered Summaries
Grok AI analyzes search results and provides intelligent summaries, making it easier to understand key information at a glance.

## Troubleshooting

### "API key not configured"
- Make sure you've created a `.env` file in the root directory
- Add your xAI API key to the `.env` file: `XAI_API_KEY=your_key_here`
- Restart the server after updating the `.env` file

### "Cannot GET /"
- Make sure the server is running with `npm start`
- Check that port 3000 is not in use by another application

### "Search failed: HTTP 401"
- Your API key is invalid or expired
- Get a new key from [xAI Console](https://console.x.ai/)
- Update the key in your `.env` file and restart the server

### "Search failed: HTTP 429"
- You've exceeded the API rate limit
- Wait a few moments and try again

### No results showing
- Try broadening your search query
- Adjust your date range
- Check that at least one source is selected

## API Documentation

For more information about the xAI API, visit:
- [xAI Documentation](https://docs.x.ai/)
- [Live Search Guide](https://docs.x.ai/docs/guides/live-search)
- [API Reference](https://docs.x.ai/api)

## License

This project is provided as-is for educational and development purposes.

## Contributing

Feel free to fork this project and customize it for your needs!

## Acknowledgments

- Powered by [xAI](https://x.ai/) Grok API
- Built with vanilla JavaScript, HTML, and CSS

