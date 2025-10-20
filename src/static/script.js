// Log to confirm script version
console.log('Loaded script.js version: 2025-10-20-4');

// Automatically detect API base URL
// If running locally, use localhost. If deployed, use the same domain
const API_BASE_URL = window.location.origin; // This will be the Railway URL when deployed
const API_KEY = ''; // Set if your API requires authentication

console.log('API Base URL:', API_BASE_URL);

// Feature names mapping
const FEATURE_NAMES = {
    'fixed_acidity': 'Fixed Acidity (g/dm³)',
    'volatile_acidity': 'Volatile Acidity (g/dm³)',
    'citric_acid': 'Citric Acid (g/dm³)',
    'residual_sugar': 'Residual Sugar (g/dm³)',
    'chlorides': 'Chlorides (g/dm³)',
    'free_sulfur_dioxide': 'Free SO₂ (mg/dm³)',
    'total_sulfur_dioxide': 'Total SO₂ (mg/dm³)',
    'density': 'Density (g/cm³)',
    'pH': 'pH',
    'sulphates': 'Sulphates (g/dm³)',
    'alcohol': 'Alcohol (% vol)',
    'type_encoded': 'Wine Type'
};

// Store last prediction for CSV download
let lastPrediction = null;
let lastFeatures = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded: Initializing form');
    checkAPIHealth();
    setupFormSubmission();
});

// Check API Health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            headers: getHeaders()
        });
        
        if (response.ok) {
            const data = await response.json();
            const statusElement = document.getElementById('modelStatus');
            
            if (data.model_loaded) {
                statusElement.textContent = '✅ Ready';
                statusElement.style.color = '#10b981';
            } else {
                statusElement.textContent = '⚠️ Model Not Loaded';
                statusElement.style.color = '#f59e0b';
            }
        } else {
            document.getElementById('modelStatus').textContent = '❌ API Offline';
            document.getElementById('modelStatus').style.color = '#ef4444';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        document.getElementById('modelStatus').textContent = '❌ Connection Failed';
        document.getElementById('modelStatus').style.color = '#ef4444';
    }
}

// Get headers for API requests
function getHeaders() {
    const headers = {
        'Content-Type': 'application/json'
    };
    
    if (API_KEY) {
        headers['x-api-key'] = API_KEY;
    }
    
    return headers;
}

// Setup form submission
function setupFormSubmission() {
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        console.log('Form submitted');
        const predictBtn = document.getElementById('predictBtn');
        const btnText = predictBtn.querySelector('.btn-text');
        const loader = predictBtn.querySelector('.loader');
        btnText.style.display = 'none';
        loader.style.display = 'inline-block';
        predictBtn.disabled = true;

        try {
            const formData = new FormData(form);
            const features = Object.fromEntries(formData);
            console.log('Raw form features:', features);
            const featureValues = Object.keys(features).map(key => {
                const value = key === 'type_encoded' ? parseInt(features[key]) : parseFloat(features[key]);
                console.log(`Parsed ${key}: ${value}`);
                return value;
            });
            
            // Check for NaN values before sending to API
            if (featureValues.some(isNaN)) {
                console.log('Invalid feature values:', featureValues);
                throw new Error('Invalid input: All fields must be valid numbers');
            }
            
            console.log('Sending feature values to API:', featureValues);
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: getHeaders(),
                body: JSON.stringify({ features: featureValues })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `API error: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('API response:', data);
            displayResults(data);
            displayFeatures(features);
        } catch (error) {
            console.error('Submission error:', error);
            showError(error.message || 'Failed to make prediction');
        } finally {
            btnText.style.display = 'inline';
            loader.style.display = 'none';
            predictBtn.disabled = false;
        }
    });
}

// Display prediction results
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const qualityValue = document.getElementById('qualityValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    
    qualityValue.textContent = data.prediction;
    confidenceValue.textContent = `${(data.probability * 100).toFixed(2)}%`;
    confidenceFill.style.width = `${data.probability * 100}%`;
    resultsSection.style.display = 'block';
    
    lastPrediction = data;
}

// Display input features
function displayFeatures(features) {
    const featuresList = document.getElementById('featuresList');
    featuresList.innerHTML = '';
    
    for (const [key, value] of Object.entries(features)) {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        featureItem.innerHTML = `
            <div class="feature-name">${FEATURE_NAMES[key]}</div>
            <div class="feature-value">${key === 'type_encoded' ? (value === '0' ? 'Red Wine' : 'White Wine') : value}</div>
        `;
        featuresList.appendChild(featureItem);
    }
    
    lastFeatures = features;
}

// Show error message
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

// Fill example red wine data
function fillExampleRed() {
    const form = document.getElementById('predictionForm');
    console.log('Filling red wine example');
    form.fixed_acidity.value = 7.4;
    form.volatile_acidity.value = 0.7;
    form.citric_acid.value = 0.0;
    form.residual_sugar.value = 1.9;
    form.chlorides.value = 0.076;
    form.free_sulfur_dioxide.value = 11;
    form.total_sulfur_dioxide.value = 34;
    form.density.value = 0.9978;
    form.pH.value = 3.51;
    form.sulphates.value = 0.56;
    form.alcohol.value = 9.4;
    form.type_encoded.value = 0;
}

// Fill example white wine data
function fillExampleWhite() {
    const form = document.getElementById('predictionForm');
    console.log('Filling white wine example');
    form.fixed_acidity.value = 7.0;
    form.volatile_acidity.value = 0.27;
    form.citric_acid.value = 0.36;
    form.residual_sugar.value = 20.7;
    form.chlorides.value = 0.045;
    form.free_sulfur_dioxide.value = 45;
    form.total_sulfur_dioxide.value = 170;
    form.density.value = 1.001;
    form.pH.value = 3.0;
    form.sulphates.value = 0.45;
    form.alcohol.value = 8.8;
    form.type_encoded.value = 1;
}

// Clear form
function clearForm() {
    const form = document.getElementById('predictionForm');
    form.reset();
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
}

// Reset prediction
function resetPrediction() {
    clearForm();
    lastPrediction = null;
    lastFeatures = null;
}

// Download results as CSV
function downloadResults() {
    if (!lastPrediction || !lastFeatures) {
        showError('No prediction available to download');
        return;
    }
    
    const csvRows = [
        ['Feature', 'Value'],
        ...Object.entries(lastFeatures).map(([key, value]) => [
            FEATURE_NAMES[key],
            key === 'type_encoded' ? (value === '0' ? 'Red Wine' : 'White Wine') : value
        ]),
        [],
        ['Prediction', lastPrediction.prediction],
        ['Probability', lastPrediction.probability],
        ['Class Name', lastPrediction.class_name]
    ];
    
    const csvContent = csvRows.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'wine_quality_prediction.csv';
    a.click();
    URL.revokeObjectURL(url);
}