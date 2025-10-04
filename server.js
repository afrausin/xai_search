require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// API endpoint to proxy xAI requests
app.post('/api/search', async (req, res) => {
    console.log('\n📥 Received search request');
    console.log('Request body:', JSON.stringify(req.body, null, 2));
    
    try {
        const { systemPrompt, query, sources, country, fromDate, toDate, maxResults } = req.body;

        // Validate API key
        if (!process.env.XAI_API_KEY) {
            return res.status(500).json({ 
                error: 'API key not configured. Please add XAI_API_KEY to your .env file' 
            });
        }

        // Validate max_search_results (xAI API requires it to be lower than 30)
        if (maxResults && maxResults >= 30) {
            return res.status(400).json({
                error: 'Max results must be lower than 30 (xAI API limitation)'
            });
        }

        // Build sources with country parameter for web and news
        const processedSources = sources.map(source => {
            // Add country parameter for web and news sources
            if (country && (source.type === 'web' || source.type === 'news')) {
                return {
                    ...source,
                    country: country
                };
            }
            return source;
        });

        // Build search parameters
        const searchParameters = {
            mode: 'on',
            sources: processedSources,
            return_citations: true  // Enable citations
        };

        if (fromDate) {
            searchParameters.from_date = fromDate;
        }

        if (toDate) {
            searchParameters.to_date = toDate;
        }

        if (maxResults) {
            searchParameters.max_search_results = maxResults;
        }

        // Log request for debugging
        console.log('Making request to xAI with sources:', JSON.stringify(processedSources, null, 2));
        console.log('Search parameters:', JSON.stringify(searchParameters, null, 2));
        
        if (fromDate || toDate) {
            console.log(`📅 Date filter active: ${fromDate || 'beginning'} → ${toDate || 'today'}`);
        }

        // Use custom system prompt or default
        const finalSystemPrompt = systemPrompt && systemPrompt.trim() 
            ? systemPrompt.trim() 
            : 'You are a search assistant. ONLY use information from the provided search results. If no relevant results are found in the specified date range, explicitly state that no information was found for that time period. Do not use your general knowledge.';

        console.log('📝 Using system prompt:', finalSystemPrompt);

        // Make request to xAI API
        const response = await fetch('https://api.x.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.XAI_API_KEY}`
            },
            body: JSON.stringify({
                model: 'grok-2-latest',
                messages: [
                    {
                        role: 'system',
                        content: finalSystemPrompt
                    },
                    {
                        role: 'user',
                        content: `Search for: ${query}`
                    }
                ],
                search_parameters: searchParameters,
                stream: false
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('xAI API Error:', response.status, errorData);
            return res.status(response.status).json({
                error: errorData.error?.message || errorData.message || `xAI API error: ${response.statusText}`
            });
        }

        const data = await response.json();
        
        // Log response structure for debugging
        console.log('\n✅ xAI API Response received');
        console.log('Top level has citations:', !!data.citations);
        console.log('Citations count:', data.citations?.length || 0);
        if (data.citations && data.citations.length > 0) {
            console.log('First 3 citations:', data.citations.slice(0, 3));
        }
        if (data.usage) {
            console.log('Sources used:', data.usage.num_sources_used);
        }
        console.log('─────────────────────────────────────\n');
        
        res.json(data);

    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({ 
            error: error.message || 'An error occurred while processing your search' 
        });
    }
});

// Load settings endpoint
app.get('/api/settings', (req, res) => {
    const fs = require('fs');
    const settingsPath = path.join(__dirname, 'batch-settings.json');
    
    try {
        if (fs.existsSync(settingsPath)) {
            const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
            res.json(settings);
        } else {
            // Return default settings if file doesn't exist
            res.json({
                queryTemplate: "Did {brand} run any promotions in {month-year}?",
                source: "x",
                country: "US",
                startMonth: "2024-01",
                endMonth: "2024-12",
                lookbackMonths: 3,
                maxResults: 20,
                enableSentiment: false,
                xFilters: {
                    includedHandles: "",
                    excludedHandles: "",
                    minFavorites: "",
                    minViews: ""
                },
                systemPrompt: "You are a search assistant. ONLY use information from the provided search results. If no relevant results are found in the specified date range, explicitly state that no information was found for that time period. Do not use your general knowledge."
            });
        }
    } catch (error) {
        console.error('Error reading settings:', error);
        res.status(500).json({ error: 'Failed to load settings' });
    }
});

// Save settings endpoint
app.post('/api/settings', (req, res) => {
    const fs = require('fs');
    const settingsPath = path.join(__dirname, 'batch-settings.json');
    
    console.log('\n📝 Saving settings to:', settingsPath);
    console.log('Settings data:', JSON.stringify(req.body, null, 2));
    
    try {
        fs.writeFileSync(settingsPath, JSON.stringify(req.body, null, 2), 'utf8');
        console.log('✅ Settings saved successfully to batch-settings.json');
        res.json({ success: true, message: 'Settings saved successfully' });
    } catch (error) {
        console.error('❌ Error saving settings:', error.message);
        console.error('Full error:', error);
        res.status(500).json({ 
            error: 'Failed to save settings', 
            details: error.message 
        });
    }
});

// Save CSV endpoint
app.post('/api/save-csv', async (req, res) => {
    try {
        const { filename, content, metadata } = req.body;
        
        // Create downloads directory if it doesn't exist
        const fs = require('fs');
        const downloadsDir = path.join(__dirname, 'downloads');
        if (!fs.existsSync(downloadsDir)) {
            fs.mkdirSync(downloadsDir);
        }
        
        // Save CSV file
        const filePath = path.join(downloadsDir, filename);
        fs.writeFileSync(filePath, content, 'utf8');
        
        // Save metadata
        const metadataPath = path.join(downloadsDir, `${filename}.meta.json`);
        fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2), 'utf8');
        
        console.log(`📊 Saved CSV: ${filename}`);
        
        res.json({ success: true, filename });
    } catch (error) {
        console.error('Error saving CSV:', error);
        res.status(500).json({ error: error.message });
    }
});

// Serve downloads directory
app.use('/downloads', express.static(path.join(__dirname, 'downloads')));

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        apiKeyConfigured: !!process.env.XAI_API_KEY 
    });
});

app.listen(PORT, () => {
    console.log(`\n🚀 xAI Search server running at http://localhost:${PORT}`);
    console.log(`📝 API Key configured: ${process.env.XAI_API_KEY ? '✅ Yes' : '❌ No - Please add to .env file'}\n`);
});

