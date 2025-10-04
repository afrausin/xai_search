// xAI Search Application
class XAISearchApp {
    constructor() {
        this.apiEndpoint = '/api/search';
        
        this.initializeElements();
        this.attachEventListeners();
        this.setDefaultDates();
        this.checkServerHealth();
    }

    initializeElements() {
        this.elements = {
            systemPrompt: document.getElementById('system-prompt'),
            searchQuery: document.getElementById('search-query'),
            sourceCheckboxes: document.querySelectorAll('input[name="source"]'),
            country: document.getElementById('country'),
            fromDate: document.getElementById('from-date'),
            toDate: document.getElementById('to-date'),
            maxResults: document.getElementById('max-results'),
            includedHandles: document.getElementById('included-handles'),
            excludedHandles: document.getElementById('excluded-handles'),
            minFavorites: document.getElementById('min-favorites'),
            minViews: document.getElementById('min-views'),
            searchBtn: document.getElementById('search-btn'),
            resultsContainer: document.getElementById('results-container'),
            results: document.getElementById('results'),
            resultsCount: document.getElementById('results-count'),
            errorContainer: document.getElementById('error-container')
        };
    }

    attachEventListeners() {
        this.elements.searchBtn.addEventListener('click', () => this.handleSearch());
        
        // Auto-resize search query textarea
        this.elements.searchQuery.addEventListener('input', () => this.autoResizeTextarea(this.elements.searchQuery));
        
        // Auto-resize system prompt textarea
        this.elements.systemPrompt.addEventListener('input', () => this.autoResizeTextarea(this.elements.systemPrompt));
        
        // Initial resize
        setTimeout(() => {
            this.autoResizeTextarea(this.elements.searchQuery);
            this.autoResizeTextarea(this.elements.systemPrompt);
        }, 0);
        
        // Allow Enter key to submit (Shift+Enter for new line)
        this.elements.searchQuery.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSearch();
            }
        });
    }
    
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }

    setDefaultDates() {
        const today = new Date();
        const thirtyDaysAgo = new Date(today);
        thirtyDaysAgo.setDate(today.getDate() - 30);

        this.elements.toDate.valueAsDate = today;
        this.elements.fromDate.valueAsDate = thirtyDaysAgo;
    }

    async checkServerHealth() {
        // Check if page is opened with file:// protocol
        if (window.location.protocol === 'file:') {
            this.showError('⚠️ Please open this app through the server at <a href="http://localhost:3000" style="color: var(--error); text-decoration: underline;">http://localhost:3000</a><br><br>Opening the HTML file directly will not work. Run <code>npm start</code> and visit http://localhost:3000');
            return;
        }

        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (!data.apiKeyConfigured) {
                this.showError('Server is running but API key is not configured. Please add your xAI API key to the .env file.');
            }
        } catch (error) {
            console.warn('Could not connect to server:', error);
            this.showError('Cannot connect to server. Make sure the server is running with <code>npm start</code>');
        }
    }

    getSelectedSources() {
        const sources = [];
        this.elements.sourceCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                const sourceObj = { type: checkbox.value };
                
                // Add X-specific filters if this is an X source
                if (checkbox.value === 'x') {
                    const includedHandles = this.elements.includedHandles.value.trim();
                    const excludedHandles = this.elements.excludedHandles.value.trim();
                    const minFavorites = this.elements.minFavorites.value.trim();
                    const minViews = this.elements.minViews.value.trim();
                    
                    if (includedHandles) {
                        sourceObj.included_x_handles = includedHandles.split(',').map(h => h.trim()).filter(h => h);
                    }
                    if (excludedHandles) {
                        sourceObj.excluded_x_handles = excludedHandles.split(',').map(h => h.trim()).filter(h => h);
                    }
                    if (minFavorites) {
                        sourceObj.post_favorite_count = parseInt(minFavorites);
                    }
                    if (minViews) {
                        sourceObj.post_view_count = parseInt(minViews);
                    }
                }
                
                sources.push(sourceObj);
            }
        });
        return sources;
    }

    validateInputs() {
        const errors = [];

        const query = this.elements.searchQuery.value.trim();
        if (!query) {
            errors.push('Please enter a search query');
        }

        const sources = this.getSelectedSources();
        if (sources.length === 0) {
            errors.push('Please select at least one search source');
        }

        const maxResults = parseInt(this.elements.maxResults.value);
        if (maxResults < 1 || maxResults > 29) {
            errors.push('Max results must be between 1 and 29');
        }

        const fromDate = this.elements.fromDate.value;
        const toDate = this.elements.toDate.value;

        if (fromDate && toDate && new Date(fromDate) > new Date(toDate)) {
            errors.push('From date cannot be after To date');
        }

        // Validate X handle filters
        const includedHandles = this.elements.includedHandles.value.trim();
        const excludedHandles = this.elements.excludedHandles.value.trim();
        
        if (includedHandles && excludedHandles) {
            errors.push('Cannot use both included and excluded X handles at the same time');
        }
        
        if (includedHandles) {
            const handles = includedHandles.split(',').map(h => h.trim()).filter(h => h);
            if (handles.length > 10) {
                errors.push('Maximum 10 included X handles allowed');
            }
        }
        
        if (excludedHandles) {
            const handles = excludedHandles.split(',').map(h => h.trim()).filter(h => h);
            if (handles.length > 10) {
                errors.push('Maximum 10 excluded X handles allowed');
            }
        }

        return { valid: errors.length === 0, errors };
    }

    showError(message) {
        this.elements.errorContainer.innerHTML = `
            <h3 style="font-weight: 500; margin-bottom: 0.5rem;">Error</h3>
            <p style="font-size: 0.9rem;">${message}</p>
        `;
        this.elements.errorContainer.style.display = 'block';
        this.elements.resultsContainer.style.display = 'none';
    }

    hideError() {
        this.elements.errorContainer.style.display = 'none';
    }

    setLoading(isLoading) {
        this.elements.searchBtn.disabled = isLoading;
        this.elements.searchBtn.querySelector('.btn-text').style.display = isLoading ? 'none' : 'inline';
        this.elements.searchBtn.querySelector('.btn-loader').style.display = isLoading ? 'inline' : 'none';
    }

    async handleSearch() {
        this.hideError();

        const validation = this.validateInputs();
        if (!validation.valid) {
            this.showError(validation.errors.join('<br>'));
            return;
        }

        const systemPrompt = this.elements.systemPrompt.value.trim();
        const query = this.elements.searchQuery.value.trim();
        const sources = this.getSelectedSources();
        const country = this.elements.country.value;
        const fromDate = this.elements.fromDate.value;
        const toDate = this.elements.toDate.value;
        const maxResults = parseInt(this.elements.maxResults.value) || 10;

        this.setLoading(true);

        try {
            const results = await this.performSearch(systemPrompt, query, sources, country, fromDate, toDate, maxResults);
            this.displayResults(results, query, sources);
        } catch (error) {
            console.error('Search error:', error);
            this.showError(`Search failed: ${error.message}`);
        } finally {
            this.setLoading(false);
        }
    }

    async performSearch(systemPrompt, query, sources, country, fromDate, toDate, maxResults) {
        try {
            console.log('Sending search request...', { systemPrompt, query, sources, country, fromDate, toDate, maxResults });
            
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    systemPrompt,
                    query,
                    sources,
                    country,
                    fromDate,
                    toDate,
                    maxResults
                })
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.error('Server error:', errorData);
                throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Search response received');
            return data;
        } catch (error) {
            console.error('Fetch error:', error);
            throw error;
        }
    }

    displayResults(data, query, sources) {
        this.elements.results.innerHTML = '';
        this.elements.resultsContainer.style.display = 'block';

        // Extract the response content
        const content = data.choices?.[0]?.message?.content || 'No results found';
        
        // Check if there are citations in the response
        const citations = data.choices?.[0]?.message?.citations || [];
        
        // Check if there are search results in the response
        const searchResults = data.choices?.[0]?.message?.search_results || [];
        
        // Display the AI summary with citations
        const summaryItem = document.createElement('div');
        summaryItem.className = 'result-item';
        
        let citationsHtml = '';
        if (citations && citations.length > 0) {
            citationsHtml = `
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);">
                    <h4 style="font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--text-secondary);">
                        Sources (${citations.length})
                    </h4>
                    <div class="citations-list">
                        ${citations.map((url, index) => `
                            <a href="${this.escapeHtml(url)}" 
                               target="_blank" 
                               rel="noopener noreferrer"
                               style="display: block; color: var(--text-primary); text-decoration: none; padding: 0.375rem 0; font-size: 0.875rem; border-bottom: 1px solid transparent; transition: border-color 0.2s;"
                               onmouseover="this.style.borderBottomColor='var(--text-primary)'"
                               onmouseout="this.style.borderBottomColor='transparent'">
                                ${index + 1}. ${this.escapeHtml(this.shortenUrl(url))}
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        summaryItem.innerHTML = `
            <div class="result-content">
                <h3 style="margin-bottom: 0.75rem; font-weight: 600; font-size: 1.125rem;">AI Summary</h3>
                <p style="white-space: pre-wrap; line-height: 1.7;">${this.escapeHtml(content)}</p>
                ${citationsHtml}
            </div>
            <div class="result-meta">
                <span class="result-badge">Grok</span>
            </div>
        `;
        this.elements.results.appendChild(summaryItem);

        // Display search results if available
        if (searchResults && searchResults.length > 0) {
            this.elements.resultsCount.textContent = `${searchResults.length} search result(s) found`;
            
            searchResults.forEach((result, index) => {
                const resultItem = this.createResultItem(result, index + 1);
                this.elements.results.appendChild(resultItem);
            });
        } else {
            this.elements.resultsCount.textContent = 'Search completed';
        }

        // Display usage information if available
        if (data.usage) {
            const usageInfo = document.createElement('div');
            usageInfo.className = 'result-item';
            usageInfo.innerHTML = `
                <div class="result-content">
                    <h4 style="margin-bottom: 0.5rem; font-weight: 500; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;">Usage Information</h4>
                    <p style="font-size: 0.875rem; color: var(--text-secondary);">
                        ${data.usage.prompt_tokens} prompt + ${data.usage.completion_tokens} completion = ${data.usage.total_tokens} total tokens
                    </p>
                </div>
            `;
            this.elements.results.appendChild(usageInfo);
        }
    }

    createResultItem(result, index) {
        const item = document.createElement('div');
        item.className = 'result-item';
        
        const title = result.title || result.text || `Result ${index}`;
        const snippet = result.snippet || result.text || '';
        const url = result.url || '';
        const source = result.source || 'unknown';
        const date = result.date || result.published_at || '';

        item.innerHTML = `
            <div class="result-content">
                <h3 style="color: var(--text-primary); margin-bottom: 0.75rem; font-weight: 600; font-size: 1.125rem;">
                    ${this.escapeHtml(title)}
                </h3>
                ${snippet ? `<p style="color: var(--text-secondary); margin-bottom: 0.75rem; line-height: 1.7;">${this.escapeHtml(snippet)}</p>` : ''}
                ${url ? `<a href="${this.escapeHtml(url)}" target="_blank" rel="noopener noreferrer" style="color: var(--text-primary); text-decoration: none; font-size: 0.875rem; border-bottom: 1px solid var(--border);">${this.escapeHtml(url)}</a>` : ''}
            </div>
            <div class="result-meta">
                ${source ? `<span class="result-badge">${this.escapeHtml(source)}</span>` : ''}
                ${date ? `<span class="result-badge">${this.formatDate(date)}</span>` : ''}
            </div>
        `;

        return item;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    shortenUrl(url) {
        try {
            const urlObj = new URL(url);
            const maxLength = 60;
            
            // Get domain and path
            let shortened = urlObj.hostname + urlObj.pathname;
            
            // Add query params if present
            if (urlObj.search) {
                shortened += '?...';
            }
            
            // Truncate if too long
            if (shortened.length > maxLength) {
                shortened = shortened.substring(0, maxLength) + '...';
            }
            
            return shortened;
        } catch (e) {
            // If URL parsing fails, just truncate the string
            return url.length > 60 ? url.substring(0, 60) + '...' : url;
        }
    }

    formatDate(dateString) {
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric' 
            });
        } catch (e) {
            return dateString;
        }
    }
}

// Initialize the app when the DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new XAISearchApp();
});

