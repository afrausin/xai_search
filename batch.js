// Batch Search Application
class BatchSearchApp {
    constructor() {
        this.apiEndpoint = '/api/search';
        this.saveEndpoint = '/api/save-csv';
        this.settingsEndpoint = '/api/settings';
        this.results = [];
        this.selectedForComparison = new Set();
        this.customFields = [];
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadSettings();
        this.loadPreviousSearches();
    }

    initializeElements() {
        this.elements = {
            query: document.getElementById('batch-query'),
            sourceRadios: document.querySelectorAll('input[name="batch-source"]'),
            country: document.getElementById('batch-country'),
            startMonth: document.getElementById('batch-start'),
            endMonth: document.getElementById('batch-end'),
            lookbackMonths: document.getElementById('lookback-months'),
            maxResults: document.getElementById('batch-results'),
            enableSentiment: document.getElementById('enable-sentiment'),
            includedHandles: document.getElementById('batch-included-handles'),
            excludedHandles: document.getElementById('batch-excluded-handles'),
            minFavorites: document.getElementById('batch-min-favorites'),
            minViews: document.getElementById('batch-min-views'),
            systemPrompt: document.getElementById('batch-system-prompt'),
            startBtn: document.getElementById('batch-start-btn'),
            testBtn: document.getElementById('batch-test-btn'),
            saveSettingsBtn: document.getElementById('save-settings-btn'),
            progressContainer: document.getElementById('batch-progress-compact'),
            resultsPreview: document.getElementById('batch-results-preview'),
            progressBar: document.getElementById('progress-bar'),
            progressText: document.getElementById('progress-text'),
            resultsList: document.getElementById('results-list'),
            previousSearches: document.getElementById('previous-searches'),
            customFieldsContainer: document.getElementById('custom-fields-container'),
            addFieldBtn: document.getElementById('add-field-btn')
        };
    }

    attachEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.startBatchSearch(false));
        this.elements.testBtn.addEventListener('click', () => this.startBatchSearch(true));
        this.elements.saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        this.elements.addFieldBtn.addEventListener('click', () => this.addCustomField());
        
        // Auto-resize batch query textarea
        this.elements.query.addEventListener('input', () => this.autoResizeTextarea(this.elements.query));
        
        // Auto-resize system prompt textarea
        this.elements.systemPrompt.addEventListener('input', () => this.autoResizeTextarea(this.elements.systemPrompt));
        
        // Initial resize
        this.autoResizeTextarea(this.elements.query);
        this.autoResizeTextarea(this.elements.systemPrompt);
    }
    
    addCustomField(name = '', value = '') {
        const fieldIndex = this.customFields.length;
        this.customFields.push({ name, value });
        
        const fieldDiv = document.createElement('div');
        fieldDiv.className = 'custom-field';
        fieldDiv.dataset.index = fieldIndex;
        fieldDiv.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr auto; gap: 0.5rem; margin-bottom: 0.5rem; align-items: start;';
        
        fieldDiv.innerHTML = `
            <div>
                <input 
                    type="text" 
                    class="field-name" 
                    placeholder="Field name (e.g., brand)"
                    value="${this.escapeHtml(name)}"
                    style="width: 100%; padding: 0.5rem; border: 1px solid var(--border); border-radius: 0.375rem; font-size: 0.875rem;"
                    title="Field name without braces. Use in templates as {field-name}"
                />
            </div>
            <div>
                <input 
                    type="text" 
                    class="field-value" 
                    placeholder="Value (e.g., Zara)"
                    value="${this.escapeHtml(value)}"
                    style="width: 100%; padding: 0.5rem; border: 1px solid var(--border); border-radius: 0.375rem; font-size: 0.875rem;"
                />
            </div>
            <button 
                type="button"
                class="remove-field-btn"
                style="padding: 0.5rem 0.75rem; background: transparent; border: 1px solid var(--border); color: var(--text-secondary); border-radius: 0.375rem; cursor: pointer; transition: all 0.2s;"
                onmouseover="this.style.borderColor='var(--error)'; this.style.color='var(--error)';"
                onmouseout="this.style.borderColor='var(--border)'; this.style.color='var(--text-secondary)';"
                title="Remove field"
            >×</button>
        `;
        
        // Add remove functionality
        fieldDiv.querySelector('.remove-field-btn').addEventListener('click', () => {
            const index = parseInt(fieldDiv.dataset.index);
            this.customFields.splice(index, 1);
            fieldDiv.remove();
            // Update indices
            this.elements.customFieldsContainer.querySelectorAll('.custom-field').forEach((div, i) => {
                div.dataset.index = i;
            });
        });
        
        // Update customFields array when inputs change
        const nameInput = fieldDiv.querySelector('.field-name');
        const valueInput = fieldDiv.querySelector('.field-value');
        
        nameInput.addEventListener('input', () => {
            this.customFields[fieldIndex].name = nameInput.value;
        });
        
        valueInput.addEventListener('input', () => {
            this.customFields[fieldIndex].value = valueInput.value;
        });
        
        this.elements.customFieldsContainer.appendChild(fieldDiv);
    }
    
    removeCustomField(index) {
        this.customFields.splice(index, 1);
        this.renderCustomFields();
    }
    
    applyReplacements(text, monthYear = null, fieldValues = null) {
        let result = text;
        
        // Replace custom fields with provided values or default values
        this.customFields.forEach(field => {
            if (field.name) {
                // Remove braces from field name if user included them
                let fieldName = field.name.trim();
                if (fieldName.startsWith('{') && fieldName.endsWith('}')) {
                    fieldName = fieldName.slice(1, -1);
                }
                
                const placeholder = `{${fieldName}}`;
                // Escape special regex characters in the placeholder
                const escapedPlaceholder = placeholder.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                
                // Use provided field value or default to the field's value
                const valueToUse = fieldValues && fieldValues[fieldName] !== undefined 
                    ? fieldValues[fieldName] 
                    : field.value;
                
                if (valueToUse) {
                    result = result.replace(new RegExp(escapedPlaceholder, 'g'), valueToUse);
                }
            }
        });
        
        // Replace {month-year} if provided
        if (monthYear) {
            result = result.replace(/{month-year}/g, monthYear);
        }
        
        return result;
    }
    
    // Generate all combinations of custom field values (for comma-separated values)
    generateFieldCombinations() {
        // Parse custom fields and split comma-separated values
        const fieldGroups = this.customFields
            .filter(f => f.name && f.value)
            .map(field => {
                let fieldName = field.name.trim();
                if (fieldName.startsWith('{') && fieldName.endsWith('}')) {
                    fieldName = fieldName.slice(1, -1);
                }
                
                // Split by comma and trim
                const values = field.value.split(',').map(v => v.trim()).filter(v => v);
                
                return {
                    name: fieldName,
                    values: values
                };
            });
        
        // If no fields or all fields have single values, return single combination
        if (fieldGroups.length === 0) {
            return [{}];
        }
        
        // Generate all combinations using Cartesian product
        function cartesianProduct(arrays) {
            if (arrays.length === 0) return [[]];
            if (arrays.length === 1) return arrays[0].map(x => [x]);
            
            const [first, ...rest] = arrays;
            const restProduct = cartesianProduct(rest);
            
            return first.flatMap(x => 
                restProduct.map(combo => [x, ...combo])
            );
        }
        
        const valueArrays = fieldGroups.map(fg => fg.values);
        const combinations = cartesianProduct(valueArrays);
        
        // Convert to objects with field names as keys
        return combinations.map(combo => {
            const obj = {};
            fieldGroups.forEach((fg, idx) => {
                obj[fg.name] = combo[idx];
            });
            return obj;
        });
    }
    
    async loadSettings() {
        try {
            const response = await fetch(this.settingsEndpoint);
            if (!response.ok) {
                console.warn('Could not load settings');
                return;
            }
            
            const settings = await response.json();
            
            // Apply settings to form
            if (settings.queryTemplate) this.elements.query.value = settings.queryTemplate;
            if (settings.source) {
                const sourceRadio = Array.from(this.elements.sourceRadios).find(r => r.value === settings.source);
                if (sourceRadio) sourceRadio.checked = true;
            }
            if (settings.country !== undefined) this.elements.country.value = settings.country;
            if (settings.startMonth) this.elements.startMonth.value = settings.startMonth;
            if (settings.endMonth) this.elements.endMonth.value = settings.endMonth;
            if (settings.lookbackMonths) this.elements.lookbackMonths.value = settings.lookbackMonths;
            if (settings.maxResults) this.elements.maxResults.value = settings.maxResults;
            if (settings.enableSentiment !== undefined) this.elements.enableSentiment.checked = settings.enableSentiment;
            if (settings.systemPrompt) this.elements.systemPrompt.value = settings.systemPrompt;
            
            // Apply X filters
            if (settings.xFilters) {
                if (settings.xFilters.includedHandles !== undefined) this.elements.includedHandles.value = settings.xFilters.includedHandles;
                if (settings.xFilters.excludedHandles !== undefined) this.elements.excludedHandles.value = settings.xFilters.excludedHandles;
                if (settings.xFilters.minFavorites !== undefined) this.elements.minFavorites.value = settings.xFilters.minFavorites;
                if (settings.xFilters.minViews !== undefined) this.elements.minViews.value = settings.xFilters.minViews;
            }
            
            // Apply custom fields
            if (settings.customFields && Array.isArray(settings.customFields)) {
                this.customFields = [];
                this.elements.customFieldsContainer.innerHTML = '';
                settings.customFields.forEach(field => {
                    this.addCustomField(field.name, field.value);
                });
            }
            
            // Trigger auto-resize
            this.autoResizeTextarea(this.elements.query);
            this.autoResizeTextarea(this.elements.systemPrompt);
            
            console.log('✅ Settings loaded from batch-settings.json');
        } catch (error) {
            console.warn('Could not load settings:', error);
        }
    }
    
    async saveSettings() {
        const settings = {
            queryTemplate: this.elements.query.value.trim(),
            source: this.getSelectedSource(),
            country: this.elements.country.value,
            startMonth: this.elements.startMonth.value,
            endMonth: this.elements.endMonth.value,
            lookbackMonths: parseInt(this.elements.lookbackMonths.value),
            maxResults: parseInt(this.elements.maxResults.value),
            enableSentiment: this.elements.enableSentiment.checked,
            xFilters: {
                includedHandles: this.elements.includedHandles.value.trim(),
                excludedHandles: this.elements.excludedHandles.value.trim(),
                minFavorites: this.elements.minFavorites.value.trim(),
                minViews: this.elements.minViews.value.trim()
            },
            customFields: this.customFields.filter(f => f.name && f.value),
            systemPrompt: this.elements.systemPrompt.value.trim()
        };
        
        try {
            console.log('Saving settings:', settings);
            
            const response = await fetch(this.settingsEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.error('Server error:', errorData);
                throw new Error(errorData.details || errorData.error || 'Failed to save settings');
            }
            
            const result = await response.json();
            
            // Show success message
            const tempMessage = document.createElement('div');
            tempMessage.style.cssText = `
                position: fixed;
                top: 2rem;
                left: 50%;
                transform: translateX(-50%);
                background: var(--success);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 0.375rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 10000;
                font-size: 0.875rem;
            `;
            tempMessage.textContent = '✓ Settings saved to batch-settings.json';
            document.body.appendChild(tempMessage);
            setTimeout(() => tempMessage.remove(), 2500);
            
            console.log('✅ Settings saved:', result);
        } catch (error) {
            console.error('Save error:', error);
            alert('Error saving settings: ' + error.message);
        }
    }
    
    autoResizeTextarea(textarea) {
        // Save scroll position of parent container
        const scrollContainer = textarea.closest('[style*="overflow-y: auto"]');
        const scrollPos = scrollContainer ? scrollContainer.scrollTop : 0;
        
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
        
        // Restore scroll position
        if (scrollContainer) {
            scrollContainer.scrollTop = scrollPos;
        }
    }

    getSelectedSource() {
        const selected = Array.from(this.elements.sourceRadios).find(r => r.checked);
        return selected ? selected.value : 'x';
    }

    generateMonthRange(startMonth, endMonth) {
        const months = [];
        const start = new Date(startMonth + '-01');
        const end = new Date(endMonth + '-01');
        
        let current = new Date(start);
        while (current <= end) {
            months.push({
                year: current.getFullYear(),
                month: current.getMonth() + 1,
                monthName: current.toLocaleString('default', { month: 'long' }),
                date: new Date(current)
            });
            current.setMonth(current.getMonth() + 1);
        }
        
        return months;
    }

    getDateRange(year, month, lookbackMonths) {
        // End date: last day of the target month
        const endDate = new Date(year, month, 0);
        
        // Start date: lookbackMonths before (beginning of that month)
        const startDate = new Date(year, month - (lookbackMonths + 1), 1);
        
        return {
            from: startDate.toISOString().split('T')[0],
            to: endDate.toISOString().split('T')[0]
        };
    }

    formatMonthYear(monthName, year) {
        return `${monthName} ${year}`;
    }

    setLoading(isLoading) {
        this.elements.startBtn.disabled = isLoading;
        this.elements.startBtn.querySelector('.btn-text').style.display = isLoading ? 'none' : 'inline';
        this.elements.startBtn.querySelector('.btn-loader').style.display = isLoading ? 'inline' : 'none';
        
        this.elements.testBtn.disabled = isLoading;
        this.elements.testBtn.querySelector('.btn-text').style.display = isLoading ? 'none' : 'inline';
        this.elements.testBtn.querySelector('.btn-loader').style.display = isLoading ? 'inline' : 'none';
    }

    async startBatchSearch(testMode = false) {
        const queryTemplate = this.elements.query.value.trim();
        const systemPrompt = this.elements.systemPrompt.value.trim() || 'You are a search assistant. ONLY use information from the provided search results. If no relevant results are found in the specified date range, explicitly state that no information was found for that time period. Do not use your general knowledge.';
        const source = this.getSelectedSource();
        const country = this.elements.country.value;
        const startMonth = this.elements.startMonth.value;
        const endMonth = this.elements.endMonth.value;
        const lookbackMonths = parseInt(this.elements.lookbackMonths.value) || 3;
        const maxResults = parseInt(this.elements.maxResults.value);
        const enableSentiment = this.elements.enableSentiment.checked;
        
        // X filters
        const includedHandles = this.elements.includedHandles.value.trim();
        const excludedHandles = this.elements.excludedHandles.value.trim();
        const minFavorites = this.elements.minFavorites.value.trim();
        const minViews = this.elements.minViews.value.trim();

        if (!queryTemplate) {
            alert('Please enter a search query template');
            return;
        }

        if (!queryTemplate.includes('{month-year}')) {
            alert('Query must include {month-year} placeholder');
            return;
        }

        if (lookbackMonths < 1 || lookbackMonths > 12) {
            alert('Lookback period must be between 1 and 12 months');
            return;
        }

        // Validate X handle filters
        if (includedHandles && excludedHandles) {
            alert('Cannot use both included and excluded X handles at the same time');
            return;
        }
        
        if (includedHandles) {
            const handles = includedHandles.split(',').map(h => h.trim()).filter(h => h);
            if (handles.length > 10) {
                alert('Maximum 10 included X handles allowed');
                return;
            }
        }
        
        if (excludedHandles) {
            const handles = excludedHandles.split(',').map(h => h.trim()).filter(h => h);
            if (handles.length > 10) {
                alert('Maximum 10 excluded X handles allowed');
                return;
            }
        }

        const xFilters = {
            includedHandles,
            excludedHandles,
            minFavorites,
            minViews
        };

        let months = this.generateMonthRange(startMonth, endMonth);
        
        // If test mode, only process the first month
        if (testMode) {
            months = [months[0]];
        }
        
        // Generate all field combinations
        const fieldCombinations = this.generateFieldCombinations();
        const totalSearches = months.length * fieldCombinations.length;
        
        console.log(`🔄 Starting batch with ${months.length} months × ${fieldCombinations.length} field combinations = ${totalSearches} total searches`);
        
        this.results = [];
        this.elements.progressContainer.style.display = 'flex';
        this.elements.resultsPreview.style.display = 'block';
        this.elements.resultsList.innerHTML = '';
        this.setLoading(true);

        let searchIndex = 0;
        
        for (let i = 0; i < months.length; i++) {
            const monthData = months[i];
            const monthYear = this.formatMonthYear(monthData.monthName, monthData.year);
            const dateRange = this.getDateRange(monthData.year, monthData.month, lookbackMonths);
            
            for (let j = 0; j < fieldCombinations.length; j++) {
                searchIndex++;
                const fieldValues = fieldCombinations[j];
                const progress = (searchIndex / totalSearches) * 100;
                
                // Create display text for current field combination
                const fieldDisplay = Object.keys(fieldValues).length > 0
                    ? ` [${Object.entries(fieldValues).map(([k, v]) => `${k}=${v}`).join(', ')}]`
                    : '';
                
                this.elements.progressBar.style.width = `${progress}%`;
                this.elements.progressText.textContent = `Processing ${monthData.monthName} ${monthData.year}${fieldDisplay} (${searchIndex}/${totalSearches})`;

                const query = this.applyReplacements(queryTemplate, monthYear, fieldValues);
                const finalSystemPrompt = this.applyReplacements(systemPrompt, monthYear, fieldValues);

                try {
                    const result = await this.performSearch(
                        finalSystemPrompt,
                        query,
                        source,
                        country,
                        dateRange.from,
                        dateRange.to,
                        maxResults,
                        xFilters
                    );

                    // Citations are at the top level of the response, not in message!
                    const debugInfo = {
                        hasCitations: !!result.citations,
                        citationsCount: result.citations?.length || 0,
                        topLevelKeys: Object.keys(result)
                    };
                    console.log(`Result for ${monthYear}${fieldDisplay}:`, debugInfo);
                    console.log('Raw citations:', result.citations);

                    const resultData = {
                        month: monthYear,
                        fieldValues: fieldValues, // Store which field values were used
                        query: query,
                        dateRange: `${dateRange.from} to ${dateRange.to}`,
                        answer: result.choices?.[0]?.message?.content || 'No response',
                        citations: result.citations || [],  // Fixed: citations are at top level!
                        searchResults: result.choices?.[0]?.message?.search_results || [],
                        sourcesUsed: result.usage?.num_sources_used || 0
                    };
                
                console.log('ResultData citations:', resultData.citations);

                // Perform sentiment analysis if enabled
                if (enableSentiment && resultData.answer && resultData.answer !== 'No response') {
                    try {
                        const sentimentData = await this.analyzeSentiment(resultData.answer, query);
                        resultData.sentiment = sentimentData.sentiment;
                        resultData.sentimentScore = sentimentData.score;
                        resultData.themes = sentimentData.themes;
                        resultData.keyInsights = sentimentData.keyInsights;
                    } catch (error) {
                        console.error(`Sentiment analysis error for ${monthYear}:`, error);
                        resultData.sentiment = 'Unknown';
                        resultData.sentimentScore = 0;
                    }
                }

                this.results.push(resultData);
                this.addResultPreview(resultData);

                    // Small delay to avoid rate limiting
                    await this.sleep(enableSentiment ? 1000 : 500);

                } catch (error) {
                    console.error(`Error for ${monthYear}${fieldDisplay}:`, error);
                    this.results.push({
                        month: monthYear,
                        fieldValues: fieldValues,
                        query: query,
                        dateRange: `${dateRange.from} to ${dateRange.to}`,
                        answer: `Error: ${error.message}`,
                        citations: [],
                        sourcesUsed: 0
                    });
                }
            } // End of field combinations loop
        } // End of months loop

        this.elements.progressText.textContent = `Completed ${totalSearches} search${totalSearches > 1 ? 'es' : ''}${testMode ? ' (Test Mode)' : ''}`;
        this.setLoading(false);
        
        // Only save results if not in test mode
        if (!testMode) {
            await this.saveResults(queryTemplate, source, country, startMonth, endMonth, lookbackMonths, enableSentiment, maxResults, xFilters);
        } else {
            // Show test completion message
            const testMessage = document.createElement('div');
            testMessage.style.cssText = `
                background: var(--surface);
                border: 1px solid var(--primary-color);
                border-radius: 0.5rem;
                padding: 1rem;
                margin-top: 1rem;
                text-align: center;
            `;
            testMessage.innerHTML = `
                <div style="color: var(--primary-color); font-weight: 600; margin-bottom: 0.5rem;">✓ Test Complete</div>
                <div style="font-size: 0.875rem; color: var(--text-secondary);">
                    Test successful! Results above show output for ${months[0].monthName} ${months[0].year}.
                    <br>Ready to run full batch search.
                </div>
            `;
            this.elements.resultsList.appendChild(testMessage);
        }
    }

    async performSearch(systemPrompt, query, source, country, fromDate, toDate, maxResults, xFilters = {}) {
        // Build source object with X filters if applicable
        const sourceObj = { type: source };
        
        if (source === 'x') {
            if (xFilters.includedHandles) {
                const handles = xFilters.includedHandles.split(',').map(h => h.trim()).filter(h => h);
                if (handles.length > 0) {
                    sourceObj.included_x_handles = handles;
                }
            }
            if (xFilters.excludedHandles) {
                const handles = xFilters.excludedHandles.split(',').map(h => h.trim()).filter(h => h);
                if (handles.length > 0) {
                    sourceObj.excluded_x_handles = handles;
                }
            }
            // Only include these if they are greater than 0 (API requirement)
            if (xFilters.minFavorites) {
                const minFavs = parseInt(xFilters.minFavorites);
                if (minFavs > 0) {
                    sourceObj.post_favorite_count = minFavs;
                }
            }
            if (xFilters.minViews) {
                const minViews = parseInt(xFilters.minViews);
                if (minViews > 0) {
                    sourceObj.post_view_count = minViews;
                }
            }
        }
        
        const response = await fetch(this.apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                systemPrompt,
                query,
                sources: [sourceObj],
                country,
                fromDate,
                toDate,
                maxResults
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        return await response.json();
    }

    async analyzeSentiment(text, context) {
        const sentimentPrompt = `Analyze the following text and provide a structured sentiment analysis.

Context: ${context}

Text to analyze:
${text}

Provide your analysis in the following JSON format:
{
  "sentiment": "positive" | "negative" | "neutral" | "mixed",
  "score": <number between -1.0 (very negative) and 1.0 (very positive)>,
  "themes": [<array of 2-3 main themes or topics mentioned>],
  "keyInsights": "<one sentence summary of the most important insight>"
}

Only respond with valid JSON, no other text.`;

        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    systemPrompt: 'You are a sentiment analysis expert. Provide accurate, objective analysis in valid JSON format.',
                    query: sentimentPrompt,
                    sources: [],
                    country: '',
                    fromDate: '',
                    toDate: '',
                    maxResults: 1
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            const content = data.choices?.[0]?.message?.content || '{}';
            
            // Try to extract JSON from the response
            const jsonMatch = content.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const parsed = JSON.parse(jsonMatch[0]);
                return {
                    sentiment: parsed.sentiment || 'neutral',
                    score: parsed.score || 0,
                    themes: parsed.themes || [],
                    keyInsights: parsed.keyInsights || ''
                };
            }
            
            throw new Error('No valid JSON found in response');
        } catch (error) {
            console.error('Sentiment analysis parsing error:', error);
            return {
                sentiment: 'neutral',
                score: 0,
                themes: [],
                keyInsights: ''
            };
        }
    }

    addResultPreview(result) {
        const div = document.createElement('div');
        div.style.cssText = 'padding: 1rem; border-bottom: 1px solid var(--border);';
        
        const sentimentBadge = result.sentiment ? `<span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.5rem; background: ${this.getSentimentColor(result.sentiment)}; border-radius: 0.75rem; font-size: 0.7rem; margin-left: 0.5rem;">${this.getSentimentEmoji(result.sentiment)} ${result.sentiment}</span>` : '';
        
        // Display field values if present
        const fieldValuesBadges = result.fieldValues && Object.keys(result.fieldValues).length > 0
            ? Object.entries(result.fieldValues)
                .map(([key, value]) => `<span style="display: inline-block; padding: 0.25rem 0.5rem; background: var(--secondary-color); color: white; border-radius: 0.75rem; font-size: 0.7rem; margin-left: 0.25rem;">${this.escapeHtml(key)}: ${this.escapeHtml(value)}</span>`)
                .join('')
            : '';
        
        div.innerHTML = `
            <h4 style="margin-bottom: 0.5rem; font-size: 0.9rem; color: var(--text-primary);">
                ${this.escapeHtml(result.month)}
                ${sentimentBadge}
                ${fieldValuesBadges}
            </h4>
            <div style="background: var(--surface); padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.875rem; line-height: 1.6; color: var(--text-primary); white-space: pre-line;">${this.escapeHtml(result.answer)}</div>
            </div>
            ${result.themes && result.themes.length > 0 ? `<div style="margin-bottom: 0.5rem;">${result.themes.map(theme => `<span style="display: inline-block; padding: 0.125rem 0.5rem; background: var(--border); border-radius: 0.75rem; font-size: 0.7rem; margin-right: 0.25rem;">${this.escapeHtml(theme)}</span>`).join('')}</div>` : ''}
            ${result.citations && result.citations.length > 0 ? `
                <div style="margin-top: 0.75rem; padding: 0.75rem; background: var(--background); border-radius: 0.5rem; border: 1px solid var(--border);">
                    <div style="font-size: 0.875rem; font-weight: 600; margin-bottom: 0.75rem; color: var(--text-primary); display: flex; align-items: center; gap: 0.5rem;">
                        <span style="background: var(--primary-color); color: white; border-radius: 0.375rem; padding: 0.25rem 0.5rem; font-size: 0.7rem;">
                            ${result.citations.length}
                        </span>
                        <span>Citation${result.citations.length !== 1 ? 's' : ''}</span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem; max-height: 200px; overflow-y: auto;">${result.citations.map((url, i) => `<a href="${this.escapeHtml(url)}" target="_blank" rel="noopener noreferrer" style="font-size: 0.75rem; color: var(--primary-color); text-decoration: none; padding: 0.5rem; background: var(--surface); border-radius: 0.375rem; border: 1px solid var(--border); transition: all 0.2s; overflow-wrap: break-word; display: block;" onmouseover="this.style.borderColor='var(--primary-color)'; this.style.background='var(--border)';" onmouseout="this.style.borderColor='var(--border)'; this.style.background='var(--surface)';"><strong style="margin-right: 0.5rem;">${i + 1}.</strong>${this.escapeHtml(url)}</a>`).join('')}</div>
                </div>
            ` : `
                <div style="margin-top: 0.75rem; padding: 0.75rem; background: var(--background); border-radius: 0.5rem; border: 1px solid var(--border); text-align: center;">
                    <div style="font-size: 0.75rem; color: var(--text-secondary); font-style: italic;">
                        No citations available
                    </div>
                </div>
            `}
            ${result.searchResults && result.searchResults.length > 0 ? `
                <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border);">
                    <details>
                        <summary style="cursor: pointer; font-size: 0.75rem; font-weight: 500; color: var(--text-secondary); padding: 0.25rem;">
                            📄 Raw Sources (${result.searchResults.length})
                        </summary>
                        <div style="margin-top: 0.5rem; max-height: 300px; overflow-y: auto;">
                            ${result.searchResults.map((source, i) => `
                                <div style="background: var(--background); border: 1px solid var(--border); border-radius: 0.375rem; padding: 0.75rem; margin-bottom: 0.5rem; font-size: 0.7rem;">
                                    <div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--primary-color);">
                                        Source ${i + 1}
                                    </div>
                                    ${source.url ? `
                                        <div style="margin-bottom: 0.25rem;">
                                            <a href="${this.escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer" style="color: var(--primary-color); text-decoration: none; word-break: break-all;">
                                                🔗 ${this.escapeHtml(source.url)}
                                            </a>
                                        </div>
                                    ` : ''}
                                    ${source.title ? `
                                        <div style="margin-bottom: 0.25rem; font-weight: 500;">
                                            ${this.escapeHtml(source.title)}
                                        </div>
                                    ` : ''}
                                    ${source.snippet ? `
                                        <div style="color: var(--text-primary); line-height: 1.4; margin-top: 0.5rem;">
                                            ${this.escapeHtml(source.snippet)}
                                        </div>
                                    ` : ''}
                                    ${source.content ? `
                                        <div style="color: var(--text-primary); line-height: 1.4; margin-top: 0.5rem;">
                                            ${this.escapeHtml(source.content)}
                                        </div>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </details>
                </div>
            ` : ''}
            <small style="color: var(--text-secondary); display: block; margin-top: 0.5rem;">
                Sources used: ${result.sourcesUsed} | Date range: ${result.dateRange}
                ${result.sentimentScore !== undefined ? ` | Sentiment: ${result.sentimentScore.toFixed(2)}` : ''}
            </small>
        `;
        this.elements.resultsList.appendChild(div);
    }

    getSentimentColor(sentiment) {
        const colors = {
            'positive': '#22c55e40',
            'negative': '#ef444440',
            'neutral': '#64748b40',
            'mixed': '#f59e0b40'
        };
        return colors[sentiment] || colors.neutral;
    }

    getSentimentEmoji(sentiment) {
        const emojis = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐',
            'mixed': '🤔'
        };
        return emojis[sentiment] || '📊';
    }

    async saveResults(queryTemplate, source, country, startMonth, endMonth, lookbackMonths, enableSentiment, maxResults, xFilters) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
        const filename = `batch_search_${timestamp}.csv`;
        
        // Get system prompt
        const systemPrompt = this.elements.systemPrompt.value.trim() || 'Default system prompt';
        
        // Collect all unique custom field names from results
        const customFieldNames = new Set();
        this.results.forEach(r => {
            if (r.fieldValues) {
                Object.keys(r.fieldValues).forEach(key => customFieldNames.add(key));
            }
        });
        const customFieldNamesArray = Array.from(customFieldNames).sort();

        // Create CSV content with all settings and optional sentiment columns
        const baseHeaders = [
            'Month', 
            'Query', 
            'Date Range', 
            'Answer',
            'Citations',
            'Sources Used',
            // Settings columns
            'Query Template',
            'System Prompt',
            'Source',
            'Country',
            'Max Results',
            'Lookback Months',
            'Sentiment Enabled',
            'X Included Handles',
            'X Excluded Handles',
            'X Min Favorites',
            'X Min Views',
            // Custom fields
            ...customFieldNamesArray.map(name => `Custom: ${name}`)
        ];
        
        const sentimentHeaders = enableSentiment 
            ? ['Sentiment', 'Sentiment Score', 'Themes', 'Key Insights']
            : [];
        
        const headers = [...baseHeaders, ...sentimentHeaders, 'Raw Sources'];
        
        const rows = this.results.map(r => {
            // Custom field values for this result
            const customFieldValues = customFieldNamesArray.map(fieldName => 
                r.fieldValues?.[fieldName] || ''
            );
            
            // Format search results as JSON string for CSV
            const rawSources = r.searchResults && r.searchResults.length > 0 
                ? JSON.stringify(r.searchResults).replace(/"/g, '""')
                : '';
            
            const baseData = [
                r.month,
                r.query,
                r.dateRange,
                r.answer.replace(/"/g, '""'), // Escape quotes
                (r.citations || []).join('; '),
                r.sourcesUsed,
                // Settings
                queryTemplate,
                systemPrompt.replace(/"/g, '""'),
                source,
                country || 'All Countries',
                maxResults,
                lookbackMonths,
                enableSentiment ? 'Yes' : 'No',
                xFilters.includedHandles || '',
                xFilters.excludedHandles || '',
                xFilters.minFavorites || '0',
                xFilters.minViews || '0',
                // Custom fields
                ...customFieldValues
            ];
            
            const sentimentData = enableSentiment ? [
                r.sentiment || 'N/A',
                r.sentimentScore !== undefined ? r.sentimentScore.toFixed(2) : 'N/A',
                (r.themes || []).join(', '),
                r.keyInsights || ''
            ] : [];
            
            return [...baseData, ...sentimentData, rawSources];
        });

        const csv = [
            headers.join(','),
            ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
        ].join('\n');

        // Save to server
        try {
            const response = await fetch(this.saveEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename,
                    content: csv,
                    metadata: {
                        queryTemplate,
                        systemPrompt,
                        source,
                        country,
                        startMonth,
                        endMonth,
                        lookbackMonths,
                        maxResults,
                        enableSentiment,
                        xFilters,
                        customFields: this.customFields.filter(f => f.name && f.value),
                        totalResults: this.results.length,
                        timestamp: new Date().toISOString()
                    }
                })
            });

            if (response.ok) {
                // Save to localStorage for listing
                this.saveToPreviousSearches({
                    filename,
                    queryTemplate,
                    systemPrompt,
                    source,
                    country,
                    startMonth,
                    endMonth,
                    lookbackMonths,
                    maxResults,
                    enableSentiment,
                    xFilters,
                    customFields: this.customFields.filter(f => f.name && f.value),
                    timestamp: new Date().toISOString(),
                    results: this.results
                });
                
                this.loadPreviousSearches();
                
                alert(`Results saved as ${filename}`);
            }
        } catch (error) {
            console.error('Error saving:', error);
            // Save to localStorage even if server save fails
            this.saveToPreviousSearches({
                filename,
                queryTemplate,
                systemPrompt,
                source,
                country,
                startMonth,
                endMonth,
                lookbackMonths,
                maxResults,
                enableSentiment,
                xFilters,
                customFields: this.customFields.filter(f => f.name && f.value),
                timestamp: new Date().toISOString(),
                results: this.results
            });
            
            this.loadPreviousSearches();
        }
    }

    downloadCSV(content, filename) {
        const blob = new Blob([content], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    saveToPreviousSearches(metadata) {
        let previous = [];
        try {
            previous = JSON.parse(localStorage.getItem('batchSearches') || '[]');
        } catch (e) {}
        
        previous.unshift(metadata);
        previous = previous.slice(0, 20); // Keep last 20
        
        localStorage.setItem('batchSearches', JSON.stringify(previous));
    }

    toggleCompareSelection(index) {
        if (this.selectedForComparison.has(index)) {
            this.selectedForComparison.delete(index);
        } else {
            this.selectedForComparison.add(index);
        }
        this.loadPreviousSearches();
    }

    async compareSelected() {
        if (this.selectedForComparison.size < 2) {
            alert('Please select at least 2 searches to compare');
            return;
        }

        try {
            const previous = JSON.parse(localStorage.getItem('batchSearches') || '[]');
            const selectedSearches = Array.from(this.selectedForComparison)
                .map(index => ({ ...previous[index], originalIndex: index }))
                .filter(search => search.results && search.results.length > 0);

            if (selectedSearches.length < 2) {
                alert('Selected searches must have results available for comparison');
                return;
            }

            this.displayComparisonModal(selectedSearches);
        } catch (e) {
            alert('Error loading comparison: ' + e.message);
        }
    }

    loadPreviousSearches() {
        let previous = [];
        try {
            previous = JSON.parse(localStorage.getItem('batchSearches') || '[]');
        } catch (e) {}

        if (previous.length === 0) {
            this.elements.previousSearches.innerHTML = `
                <div style="text-align: center; padding: 3rem 1rem; color: var(--text-secondary);">
                    <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">📊</div>
                    <p>No previous searches found</p>
                    <small>Your batch searches will appear here</small>
                </div>
            `;
            return;
        }

        // Show compare button if searches are selected
        const compareButtonHtml = this.selectedForComparison.size > 0 ? `
            <div style="background: var(--surface); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border: 2px solid var(--primary-color); display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>${this.selectedForComparison.size} search${this.selectedForComparison.size !== 1 ? 'es' : ''} selected for comparison</strong>
                    <button 
                        onclick="batchSearchApp.selectedForComparison.clear(); batchSearchApp.loadPreviousSearches();"
                        style="margin-left: 1rem; padding: 0.25rem 0.75rem; background: transparent; border: 1px solid var(--border); color: var(--text-secondary); border-radius: 0.25rem; font-size: 0.875rem; cursor: pointer;"
                    >
                        Clear Selection
                    </button>
                </div>
                <button 
                    onclick="batchSearchApp.compareSelected()"
                    style="padding: 0.5rem 1.5rem; background: var(--primary-color); color: white; border: none; border-radius: 0.375rem; font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: opacity 0.2s;"
                    onmouseover="this.style.opacity='0.8';"
                    onmouseout="this.style.opacity='1';"
                >
                    Compare Selected
                </button>
            </div>
        ` : '';

        this.elements.previousSearches.innerHTML = compareButtonHtml + previous.map((search, index) => `
            <div class="batch-search-card" style="background: var(--surface); padding: 1.5rem; margin-bottom: 1rem; border-radius: 0.75rem; box-shadow: 0 2px 8px var(--shadow); border: ${this.selectedForComparison.has(index) ? '2px solid var(--primary-color)' : '1px solid var(--border)'}; position: relative;">
                ${search.results && search.results.length > 0 ? `
                <div style="position: absolute; top: 1rem; left: 1rem;">
                    <input 
                        type="checkbox" 
                        id="compare-${index}"
                        ${this.selectedForComparison.has(index) ? 'checked' : ''}
                        onchange="batchSearchApp.toggleCompareSelection(${index})"
                        style="width: 1.25rem; height: 1.25rem; cursor: pointer;"
                        title="Select for comparison"
                    />
                </div>
                ` : ''}
                <button 
                    onclick="batchSearchApp.deleteSearch(${index})"
                    style="position: absolute; top: 1rem; right: 1rem; background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 1.25rem; padding: 0.25rem 0.5rem; border-radius: 0.25rem; transition: all 0.2s;"
                    onmouseover="this.style.background='var(--border)'; this.style.color='var(--error)';"
                    onmouseout="this.style.background='none'; this.style.color='var(--text-secondary)';"
                    title="Delete this search"
                >
                    ×
                </button>
                
                <div style="margin-bottom: 1rem;">
                    <h4 style="margin-bottom: 0.5rem; font-size: 1rem; font-weight: 600; color: var(--text-primary); padding-right: 2rem; ${search.results && search.results.length > 0 ? 'padding-left: 2.5rem;' : ''}">
                        ${this.escapeHtml(search.queryTemplate)}
                    </h4>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem;">
                        <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem; font-weight: 500; text-transform: uppercase;">
                            <span style="font-size: 0.875rem;">🔍</span> ${search.source}
                        </span>
                        ${search.country ? `
                        <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem;">
                            <span style="font-size: 0.875rem;">🌍</span> ${search.country}
                        </span>
                        ` : ''}
                        ${search.enableSentiment ? `
                        <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: #22c55e40; border-radius: 1rem; font-size: 0.75rem; color: #22c55e;">
                            <span style="font-size: 0.875rem;">📊</span> Sentiment
                        </span>
                        ` : ''}
                        <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem;">
                            <span style="font-size: 0.875rem;">📅</span> ${search.startMonth} → ${search.endMonth}
                        </span>
                        <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem;">
                            <span style="font-size: 0.875rem;">⏱️</span> ${search.lookbackMonths || 3}mo lookback
                        </span>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 1rem; border-top: 1px solid var(--border); gap: 0.75rem;">
                    <small style="color: var(--text-secondary);">
                        ${this.formatRelativeTime(search.timestamp)}
                    </small>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        <button
                            onclick="batchSearchApp.loadSearchSettings(${index})"
                            style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; background: var(--surface-light); color: var(--text-primary); border: 1px solid var(--border); border-radius: 0.375rem; font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: all 0.2s;"
                            onmouseover="this.style.background='var(--border)';"
                            onmouseout="this.style.background='var(--surface-light)';"
                            title="Load these settings into the form"
                        >
                            <span>Load Settings</span>
                            <span>⚙️</span>
                        </button>
                        ${search.results && search.results.length > 0 ? `
                            <button
                                onclick="batchSearchApp.viewResults(${index})"
                                style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; background: var(--text-primary); color: white; border: none; border-radius: 0.375rem; font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: opacity 0.2s;"
                                onmouseover="this.style.opacity='0.8';"
                                onmouseout="this.style.opacity='1';"
                            >
                                <span>View Results</span>
                                <span>👁</span>
                            </button>
                        ` : ''}
                        <a 
                            href="/downloads/${search.filename}" 
                            download 
                            style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; background: var(--primary-color); color: white; text-decoration: none; border-radius: 0.375rem; font-size: 0.875rem; font-weight: 500; transition: opacity 0.2s;"
                            onmouseover="this.style.opacity='0.8';"
                            onmouseout="this.style.opacity='1';"
                        >
                            <span>Download CSV</span>
                            <span>↓</span>
                        </a>
                    </div>
                </div>
            </div>
        `).join('');
    }

    formatRelativeTime(timestamp) {
        const now = new Date();
        const then = new Date(timestamp);
        const diffMs = now - then;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
        if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        
        return then.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            year: now.getFullYear() !== then.getFullYear() ? 'numeric' : undefined 
        });
    }

    loadSearchSettings(index) {
        try {
            const previous = JSON.parse(localStorage.getItem('batchSearches') || '[]');
            const search = previous[index];
            
            if (!search) {
                alert('Search not found');
                return;
            }

            // Load basic settings
            this.elements.query.value = search.queryTemplate || '';
            
            // Set source radio button
            if (search.source) {
                const sourceRadio = Array.from(this.elements.sourceRadios).find(r => r.value === search.source);
                if (sourceRadio) sourceRadio.checked = true;
            }
            
            // Set country
            if (search.country !== undefined) {
                this.elements.country.value = search.country;
            }
            
            // Set date range
            if (search.startMonth) this.elements.startMonth.value = search.startMonth;
            if (search.endMonth) this.elements.endMonth.value = search.endMonth;
            
            // Set lookback
            if (search.lookbackMonths) {
                this.elements.lookbackMonths.value = search.lookbackMonths;
            }
            
            // Set max results
            if (search.maxResults) {
                this.elements.maxResults.value = search.maxResults;
            }
            
            // Set sentiment analysis
            if (search.enableSentiment !== undefined) {
                this.elements.enableSentiment.checked = search.enableSentiment;
            }
            
            // Set X filters
            if (search.xFilters) {
                this.elements.includedHandles.value = search.xFilters.includedHandles || '';
                this.elements.excludedHandles.value = search.xFilters.excludedHandles || '';
                this.elements.minFavorites.value = search.xFilters.minFavorites || '';
                this.elements.minViews.value = search.xFilters.minViews || '';
            }
            
            // Trigger auto-resize for query textarea
            this.autoResizeTextarea(this.elements.query);
            
            // Scroll to top of form
            this.elements.query.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Show success message
            const tempMessage = document.createElement('div');
            tempMessage.style.cssText = `
                position: fixed;
                top: 2rem;
                left: 50%;
                transform: translateX(-50%);
                background: var(--success);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 0.375rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 10000;
                font-size: 0.875rem;
            `;
            tempMessage.textContent = '✓ Settings loaded successfully';
            document.body.appendChild(tempMessage);
            setTimeout(() => tempMessage.remove(), 2000);
            
        } catch (e) {
            alert('Error loading settings: ' + e.message);
        }
    }

    deleteSearch(index) {
        if (!confirm('Are you sure you want to delete this batch search?')) {
            return;
        }

        try {
            let previous = JSON.parse(localStorage.getItem('batchSearches') || '[]');
            previous.splice(index, 1);
            localStorage.setItem('batchSearches', JSON.stringify(previous));
            this.loadPreviousSearches();
        } catch (e) {
            alert('Error deleting search: ' + e.message);
        }
    }

    clearAllSearches() {
        if (!confirm('Are you sure you want to delete ALL batch searches? This cannot be undone.')) {
            return;
        }

        try {
            localStorage.removeItem('batchSearches');
            this.loadPreviousSearches();
        } catch (e) {
            alert('Error clearing searches: ' + e.message);
        }
    }

    viewResults(index) {
        try {
            const previous = JSON.parse(localStorage.getItem('batchSearches') || '[]');
            const search = previous[index];
            
            if (!search || !search.results) {
                alert('No results available for this search');
                return;
            }

            this.displayResultsModal(search);
        } catch (e) {
            alert('Error loading results: ' + e.message);
        }
    }

    displayComparisonModal(searches) {
        // Align results by month
        const allMonths = new Set();
        searches.forEach(search => {
            search.results.forEach(result => {
                allMonths.add(result.month);
            });
        });

        // Sort months chronologically
        const sortedMonths = Array.from(allMonths).sort((a, b) => {
            const parseMonth = (monthStr) => {
                const [month, year] = monthStr.split(' ');
                const monthIndex = new Date(Date.parse(month + " 1, 2000")).getMonth();
                return new Date(year, monthIndex);
            };
            return parseMonth(a) - parseMonth(b);
        });

        // Create modal
        const modal = document.createElement('div');
        modal.id = 'comparison-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 2rem;
            overflow-y: auto;
        `;

        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            background: var(--background);
            border-radius: 0.75rem;
            max-width: 95vw;
            width: 100%;
            max-height: 90vh;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
        `;

        // Header
        const header = `
            <div style="padding: 1.5rem; border-bottom: 1px solid var(--border); background: var(--background); position: sticky; top: 0; z-index: 10;">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 1rem;">
                    <div>
                        <h2 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">
                            Batch Search Comparison
                        </h2>
                        <p style="font-size: 0.875rem; color: var(--text-secondary);">
                            Comparing ${searches.length} searches across ${sortedMonths.length} months
                        </p>
                    </div>
                    <button 
                        onclick="document.getElementById('comparison-modal').remove()"
                        style="background: none; border: none; font-size: 2rem; cursor: pointer; color: var(--text-secondary); line-height: 1; padding: 0; transition: color 0.2s;"
                        onmouseover="this.style.color='var(--text-primary)';"
                        onmouseout="this.style.color='var(--text-secondary)';"
                    >
                        ×
                    </button>
                </div>
            </div>
        `;

        // Search settings section
        const settingsSection = `
            <div style="padding: 1.5rem; border-bottom: 2px solid var(--border); background: var(--surface);">
                <h3 style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: var(--text-primary);">Search Settings</h3>
                <div style="display: grid; grid-template-columns: repeat(${searches.length}, 1fr); gap: 1rem;">
                    ${searches.map((search, idx) => `
                        <div style="background: var(--background); border: 2px solid var(--border); border-radius: 0.5rem; padding: 1rem;">
                            <div style="font-size: 0.875rem; font-weight: 600; margin-bottom: 0.75rem; color: var(--primary-color);">
                                Search ${idx + 1}
                            </div>
                            <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size: 0.75rem;">
                                <div>
                                    <div style="color: var(--text-secondary); font-weight: 500;">Query:</div>
                                    <div style="color: var(--text-primary); margin-top: 0.125rem; word-break: break-word;">
                                        ${this.escapeHtml(search.queryTemplate)}
                                    </div>
                                </div>
                                <div style="display: grid; grid-template-columns: auto 1fr; gap: 0.5rem;">
                                    <div style="color: var(--text-secondary); font-weight: 500;">Source:</div>
                                    <div style="color: var(--text-primary); text-transform: uppercase;">${search.source}</div>
                                    
                                    ${search.country ? `
                                        <div style="color: var(--text-secondary); font-weight: 500;">Country:</div>
                                        <div style="color: var(--text-primary);">${search.country}</div>
                                    ` : ''}
                                    
                                    <div style="color: var(--text-secondary); font-weight: 500;">Period:</div>
                                    <div style="color: var(--text-primary);">${search.startMonth} to ${search.endMonth}</div>
                                    
                                    <div style="color: var(--text-secondary); font-weight: 500;">Lookback:</div>
                                    <div style="color: var(--text-primary);">${search.lookbackMonths || 3} months</div>
                                    
                                    ${search.maxResults ? `
                                        <div style="color: var(--text-secondary); font-weight: 500;">Max Results:</div>
                                        <div style="color: var(--text-primary);">${search.maxResults}/month</div>
                                    ` : ''}
                                    
                                    ${search.enableSentiment ? `
                                        <div style="color: var(--text-secondary); font-weight: 500;">Features:</div>
                                        <div style="color: #22c55e;">✓ Sentiment Analysis</div>
                                    ` : ''}
                                </div>
                                ${search.xFilters && (search.xFilters.includedHandles || search.xFilters.excludedHandles || search.xFilters.minFavorites || search.xFilters.minViews) ? `
                                    <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--border);">
                                        <div style="color: var(--text-secondary); font-weight: 500; margin-bottom: 0.25rem;">X Filters:</div>
                                        ${search.xFilters.includedHandles ? `
                                            <div style="color: var(--text-primary);">Include: ${this.escapeHtml(search.xFilters.includedHandles)}</div>
                                        ` : ''}
                                        ${search.xFilters.excludedHandles ? `
                                            <div style="color: var(--text-primary);">Exclude: ${this.escapeHtml(search.xFilters.excludedHandles)}</div>
                                        ` : ''}
                                        ${search.xFilters.minFavorites ? `
                                            <div style="color: var(--text-primary);">Min Favorites: ${search.xFilters.minFavorites}</div>
                                        ` : ''}
                                        ${search.xFilters.minViews ? `
                                            <div style="color: var(--text-primary);">Min Views: ${search.xFilters.minViews}</div>
                                        ` : ''}
                                    </div>
                                ` : ''}
                                <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--border); color: var(--text-secondary); font-size: 0.7rem;">
                                    ${this.formatRelativeTime(search.timestamp)}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        // Search headers
        const searchHeaders = `
            <div style="display: grid; grid-template-columns: 180px repeat(${searches.length}, 1fr); gap: 1rem; padding: 1rem 1.5rem; background: var(--surface); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 1;">
                <div style="font-weight: 600; font-size: 0.875rem; color: var(--text-secondary);">Month</div>
                ${searches.map((search, idx) => `
                    <div style="min-width: 0; text-align: center;">
                        <div style="font-size: 0.875rem; font-weight: 600; color: var(--primary-color);">
                            Search ${idx + 1}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        // Results rows
        const resultsRows = sortedMonths.map(month => {
            const monthResults = searches.map(search => {
                return search.results.find(r => r.month === month);
            });

            return `
                <div style="display: grid; grid-template-columns: 180px repeat(${searches.length}, 1fr); gap: 1rem; padding: 1.5rem; border-bottom: 1px solid var(--border); background: var(--background);">
                    <div style="font-weight: 600; color: var(--primary-color); font-size: 1rem; align-self: start; position: sticky; left: 0; background: var(--background); padding: 0.5rem; border-radius: 0.375rem; border: 1px solid var(--border);">
                        ${this.escapeHtml(month)}
                    </div>
                    ${monthResults.map(result => {
                        if (!result) {
                            return `
                                <div style="background: var(--surface); padding: 1.5rem; border-radius: 0.5rem; min-width: 0; border: 1px solid var(--border);">
                                    <div style="text-align: center; color: var(--text-secondary); font-size: 0.875rem; font-style: italic;">
                                        No data
                                    </div>
                                </div>
                            `;
                        }
                        return `
                            <div style="background: var(--surface); padding: 1rem; border-radius: 0.5rem; min-width: 0; border: 1px solid var(--border);">
                                <div style="margin-bottom: 0.75rem; display: flex; gap: 0.25rem; flex-wrap: wrap; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);">
                                    <span style="padding: 0.25rem 0.5rem; background: var(--background); border: 1px solid var(--border); border-radius: 0.75rem; font-size: 0.7rem; font-weight: 500;">
                                        📊 ${result.sourcesUsed} source${result.sourcesUsed !== 1 ? 's' : ''}
                                    </span>
                                    ${result.sentiment ? `
                                    <span style="padding: 0.25rem 0.5rem; background: ${this.getSentimentColor(result.sentiment)}; border-radius: 0.75rem; font-size: 0.7rem; font-weight: 500;">
                                        ${this.getSentimentEmoji(result.sentiment)} ${result.sentiment}
                                    </span>
                                    ` : ''}
                                    ${result.dateRange ? `
                                    <span style="padding: 0.25rem 0.5rem; background: var(--background); border: 1px solid var(--border); border-radius: 0.75rem; font-size: 0.65rem;">
                                        📅 ${result.dateRange}
                                    </span>
                                    ` : ''}
                                </div>
                                ${result.themes && result.themes.length > 0 ? `
                                    <div style="margin-bottom: 0.75rem; display: flex; gap: 0.25rem; flex-wrap: wrap;">
                                        ${result.themes.map(theme => `
                                            <span style="padding: 0.25rem 0.5rem; background: var(--background); border: 1px solid var(--border); border-radius: 0.75rem; font-size: 0.65rem;">
                                                ${this.escapeHtml(theme)}
                                            </span>
                                        `).join('')}
                                    </div>
                                ` : ''}
                                ${result.keyInsights ? `
                                    <div style="font-size: 0.75rem; color: var(--text-primary); margin-bottom: 0.75rem; font-style: italic; padding: 0.5rem; background: var(--background); border-left: 3px solid var(--primary-color); border-radius: 0.25rem;">
                                        💡 ${this.escapeHtml(result.keyInsights)}
                                    </div>
                                ` : ''}
                                <div style="font-size: 0.8rem; line-height: 1.6; color: var(--text-primary); margin-bottom: 0.75rem; padding: 0.75rem; background: var(--background); border-radius: 0.375rem; max-height: 300px; overflow-y: auto; word-wrap: break-word; white-space: pre-wrap;">
                                    ${this.escapeHtml(result.answer)}
                                </div>
                                ${result.citations && result.citations.length > 0 ? `
                                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border);">
                                        <details>
                                            <summary style="cursor: pointer; font-size: 0.75rem; color: var(--text-secondary); font-weight: 600; padding: 0.25rem;">
                                                🔗 ${result.citations.length} citation${result.citations.length !== 1 ? 's' : ''}
                                            </summary>
                                            <div style="margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.375rem; padding: 0.5rem; background: var(--background); border-radius: 0.25rem;">
                                                ${result.citations.map((url, i) => `
                                                    <a href="${this.escapeHtml(url)}" 
                                                       target="_blank" 
                                                       rel="noopener noreferrer"
                                                       style="font-size: 0.7rem; color: var(--primary-color); text-decoration: none; overflow: hidden; text-overflow: ellipsis; display: block; padding: 0.25rem; border-radius: 0.25rem; transition: background 0.2s;"
                                                       onmouseover="this.style.background='var(--border)';"
                                                       onmouseout="this.style.background='transparent';"
                                                       title="${this.escapeHtml(url)}">
                                                        ${i + 1}. ${this.escapeHtml(url.length > 60 ? url.substring(0, 60) + '...' : url)}
                                                    </a>
                                                `).join('')}
                                            </div>
                                        </details>
                                    </div>
                                ` : ''}
                                ${result.searchResults && result.searchResults.length > 0 ? `
                                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border);">
                                        <details>
                                            <summary style="cursor: pointer; font-size: 0.75rem; color: var(--text-secondary); font-weight: 600; padding: 0.25rem;">
                                                📄 Raw Sources (${result.searchResults.length})
                                            </summary>
                                            <div style="margin-top: 0.5rem; max-height: 250px; overflow-y: auto;">
                                                ${result.searchResults.map((source, i) => `
                                                    <div style="background: var(--background); border: 1px solid var(--border); border-radius: 0.375rem; padding: 0.5rem; margin-bottom: 0.5rem; font-size: 0.65rem;">
                                                        <div style="font-weight: 600; margin-bottom: 0.25rem; color: var(--primary-color);">
                                                            Source ${i + 1}
                                                        </div>
                                                        ${source.url ? `
                                                            <div style="margin-bottom: 0.25rem;">
                                                                <a href="${this.escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer" style="color: var(--primary-color); text-decoration: none; word-break: break-all;">
                                                                    🔗 ${this.escapeHtml(source.url.length > 50 ? source.url.substring(0, 50) + '...' : source.url)}
                                                                </a>
                                                            </div>
                                                        ` : ''}
                                                        ${source.title ? `
                                                            <div style="margin-bottom: 0.25rem; font-weight: 500;">
                                                                ${this.escapeHtml(source.title)}
                                                            </div>
                                                        ` : ''}
                                                        ${source.snippet ? `
                                                            <div style="color: var(--text-primary); line-height: 1.3; margin-top: 0.25rem;">
                                                                ${this.escapeHtml(source.snippet.substring(0, 150))}${source.snippet.length > 150 ? '...' : ''}
                                                            </div>
                                                        ` : ''}
                                                        ${source.content && !source.snippet ? `
                                                            <div style="color: var(--text-primary); line-height: 1.3; margin-top: 0.25rem;">
                                                                ${this.escapeHtml(source.content.substring(0, 150))}${source.content.length > 150 ? '...' : ''}
                                                            </div>
                                                        ` : ''}
                                                    </div>
                                                `).join('')}
                                            </div>
                                        </details>
                                    </div>
                                ` : ''}
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        }).join('');

        const scrollableContent = `
            <div style="overflow-y: auto; overflow-x: auto; flex: 1;">
                ${settingsSection}
                ${searchHeaders}
                ${resultsRows}
            </div>
        `;

        modalContent.innerHTML = header + scrollableContent;
        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });

        // Close on Escape key
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }

    displayResultsModal(search) {
        // Create modal overlay
        const modal = document.createElement('div');
        modal.id = 'results-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 2rem;
            overflow-y: auto;
        `;

        // Create modal content
        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            background: var(--background);
            border-radius: 0.75rem;
            max-width: 1200px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        `;

        // Header
        const header = `
            <div style="padding: 1.5rem; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--background); z-index: 1;">
                <div style="display: flex; justify-content: space-between; align-items: start; gap: 1rem;">
                    <div>
                        <h2 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">
                            ${this.escapeHtml(search.queryTemplate)}
                        </h2>
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem;">
                                🔍 ${search.source}
                            </span>
                            ${search.country ? `
                            <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem;">
                                🌍 ${search.country}
                            </span>
                            ` : ''}
                            <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem;">
                                📅 ${search.startMonth} → ${search.endMonth}
                            </span>
                            <span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem;">
                                📊 ${search.results.length} results
                            </span>
                        </div>
                    </div>
                    <button 
                        onclick="document.getElementById('results-modal').remove()"
                        style="background: none; border: none; font-size: 2rem; cursor: pointer; color: var(--text-secondary); line-height: 1; padding: 0; transition: color 0.2s;"
                        onmouseover="this.style.color='var(--text-primary)';"
                        onmouseout="this.style.color='var(--text-secondary)';"
                    >
                        ×
                    </button>
                </div>
            </div>
        `;

        // Results
        const results = search.results.map(result => `
            <div style="padding: 1.5rem; border-bottom: 1px solid var(--border);">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <h3 style="font-size: 1rem; font-weight: 600; color: var(--primary-color);">
                        ${this.escapeHtml(result.month)}
                    </h3>
                    <span style="padding: 0.25rem 0.75rem; background: var(--border); border-radius: 1rem; font-size: 0.75rem;">
                        ${result.sourcesUsed} source${result.sourcesUsed !== 1 ? 's' : ''}
                    </span>
                </div>
                
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                        <strong>Query:</strong> ${this.escapeHtml(result.query)}
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">
                        <strong>Date Range:</strong> ${result.dateRange}
                    </div>
                </div>
                
                <div style="background: var(--surface); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--text-secondary);">
                        Answer:
                    </div>
                    <div style="white-space: pre-wrap; line-height: 1.6; color: var(--text-primary);">
                        ${this.escapeHtml(result.answer)}
                    </div>
                </div>
                
                ${result.citations && result.citations.length > 0 ? `
                    <div>
                        <div style="font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--text-secondary);">
                            Citations (${result.citations.length}):
                        </div>
                        <div style="display: flex; flex-direction: column; gap: 0.25rem;">
                            ${result.citations.map((url, i) => `
                                <a href="${this.escapeHtml(url)}" 
                                   target="_blank" 
                                   rel="noopener noreferrer"
                                   style="font-size: 0.75rem; color: var(--primary-color); text-decoration: none; padding: 0.25rem 0; border-bottom: 1px solid transparent; transition: border-color 0.2s;"
                                   onmouseover="this.style.borderBottomColor='var(--primary-color)';"
                                   onmouseout="this.style.borderBottomColor='transparent';">
                                    ${i + 1}. ${this.escapeHtml(url)}
                                </a>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                ${result.searchResults && result.searchResults.length > 0 ? `
                    <div style="margin-top: 1rem;">
                        <details>
                            <summary style="cursor: pointer; font-size: 0.875rem; font-weight: 500; color: var(--text-secondary); padding: 0.5rem; background: var(--surface); border-radius: 0.375rem;">
                                📄 Raw Sources (${result.searchResults.length})
                            </summary>
                            <div style="margin-top: 1rem; max-height: 400px; overflow-y: auto;">
                                ${result.searchResults.map((source, i) => `
                                    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 0.5rem; padding: 1rem; margin-bottom: 0.75rem;">
                                        <div style="font-weight: 600; margin-bottom: 0.75rem; color: var(--primary-color); font-size: 0.875rem;">
                                            Source ${i + 1}
                                        </div>
                                        ${source.url ? `
                                            <div style="margin-bottom: 0.5rem;">
                                                <a href="${this.escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer" style="color: var(--primary-color); text-decoration: none; font-size: 0.8rem; word-break: break-all;">
                                                    🔗 ${this.escapeHtml(source.url)}
                                                </a>
                                            </div>
                                        ` : ''}
                                        ${source.title ? `
                                            <div style="margin-bottom: 0.5rem; font-weight: 500; font-size: 0.875rem;">
                                                ${this.escapeHtml(source.title)}
                                            </div>
                                        ` : ''}
                                        ${source.snippet ? `
                                            <div style="color: var(--text-primary); line-height: 1.5; font-size: 0.8rem; margin-top: 0.5rem; padding: 0.75rem; background: var(--background); border-radius: 0.375rem;">
                                                ${this.escapeHtml(source.snippet)}
                                            </div>
                                        ` : ''}
                                        ${source.content ? `
                                            <div style="color: var(--text-primary); line-height: 1.5; font-size: 0.8rem; margin-top: 0.5rem; padding: 0.75rem; background: var(--background); border-radius: 0.375rem;">
                                                ${this.escapeHtml(source.content)}
                                            </div>
                                        ` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </details>
                    </div>
                ` : ''}
            </div>
        `).join('');

        modalContent.innerHTML = header + results;
        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });

        // Close on Escape key
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the app
let batchSearchApp;
document.addEventListener('DOMContentLoaded', () => {
    batchSearchApp = new BatchSearchApp();
});

