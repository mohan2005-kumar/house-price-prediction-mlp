/**
 * app.js
 * Dynamic frontend interactions, live real-time model inference,
 * cascading geographic dropdowns (State -> City -> Locality),
 * Month & Year Built possession controls (up to 2026),
 * slider event bindings, and UI updates for ProphetMLP.
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Core DOM Element References
  const form = document.getElementById('predictionForm');
  const priceOutput = document.getElementById('priceOutput');
  const priceRangeOutput = document.getElementById('priceRangeOutput');
  const sqftOutput = document.getElementById('sqftOutput');
  const ageOutput = document.getElementById('ageOutput');
  const amenitiesCountOutput = document.getElementById('amenitiesCountOutput');
  const priceShortBadge = document.getElementById('priceShortBadge');
  const resetBtn = document.getElementById('resetBtn');

  const stateSelect = document.getElementById('state');
  const citySelect = document.getElementById('city');
  const localitySelect = document.getElementById('locality');
  const monthSelect = document.getElementById('month_built');
  const propTypeSelect = document.getElementById('property_type');
  const furnishingSelect = document.getElementById('furnishing_status');

  // 2. Global State Variables (Declared at top of scope to avoid TDZ)
  let debounceTimeout = null;
  let currentDisplayedPrice = 0;

  // 3. Indian Currency and Number Formatting Helpers
  function formatIndianNum(num) {
    if (isNaN(num) || num === null || num === undefined) return '0';
    return Math.round(num).toLocaleString('en-IN');
  }

  function triggerDebouncedInference() {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(fetchPrediction, 80);
  }

  // Smooth number ticker animation in Indian notation
  function animatePriceNumber(start, end) {
    const s = Number(start) || 0;
    const e = Number(end) || 0;
    const duration = 280;
    const startTime = performance.now();

    function step(currentTime) {
      const progress = Math.min((currentTime - startTime) / duration, 1);
      const easeOutQuad = 1 - (1 - progress) * (1 - progress);
      const value = s + (e - s) * easeOutQuad;
      if (priceOutput) {
        priceOutput.textContent = formatIndianNum(value);
      }

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        if (priceOutput) {
          priceOutput.textContent = formatIndianNum(e);
        }
      }
    }
    requestAnimationFrame(step);
  }

  // Dynamic Property Age Display with Month Precision
  function updatePropertyAgeDisplay() {
    const yearInput = document.getElementById('year_built');
    if (!yearInput || !ageOutput) return;
    const year = parseInt(yearInput.value, 10) || 2024;
    const month = monthSelect ? monthSelect.value : 'October';
    const monthMap = {
      'January': 1, 'February': 2, 'March': 3, 'April': 4,
      'May': 5, 'June': 6, 'July': 7, 'August': 8,
      'September': 9, 'October': 10, 'November': 11, 'December': 12
    };
    const monthNum = monthMap[month] || 10;
    const age = (2026.0 - year) + (10.0 - monthNum) / 12.0;

    if (age <= 0.08) {
      ageOutput.textContent = 'Brand New / Possession Ready';
    } else if (age < 1.0) {
      const months = Math.max(1, Math.round(age * 12));
      ageOutput.textContent = `${months} month${months === 1 ? '' : 's'} (Recent)`;
    } else if (age < 2.0) {
      ageOutput.textContent = `${age.toFixed(1)} years (Recent)`;
    } else {
      ageOutput.textContent = `${Math.round(age)} years`;
    }
  }

  // 4. API Request Handler
  async function fetchPrediction() {
    if (!form) return;

    const checkedAmenities = Array.from(
      form.querySelectorAll('input[name="amenity"]:checked')
    ).map(cb => cb.value);

    const payload = {
      state: stateSelect ? stateSelect.value : 'Telangana',
      city: citySelect ? citySelect.value : 'Hyderabad',
      locality: localitySelect ? localitySelect.value : 'HITEC City & Cyber Towers',
      property_type: propTypeSelect ? propTypeSelect.value : 'Apartment / Gated High-Rise Flat',
      furnishing_status: furnishingSelect ? furnishingSelect.value : 'Semi-Furnished',
      month_built: monthSelect ? monthSelect.value : 'October',
      year_built: parseFloat(document.getElementById('year_built')?.value) || 2024,
      area_sqft: parseFloat(document.getElementById('area_sqft')?.value) || 2200,
      bedrooms: parseFloat(document.getElementById('bedrooms')?.value) || 3,
      bathrooms: parseFloat(document.getElementById('bathrooms')?.value) || 2.5,
      garage_spaces: parseFloat(document.getElementById('garage_spaces')?.value) || 2,
      distance_to_city_center_km: parseFloat(document.getElementById('distance_to_city_center_km')?.value) || 8.5,
      school_rating: parseFloat(document.getElementById('school_rating')?.value) || 8.0,
      safety_index: parseFloat(document.getElementById('safety_index')?.value) || 82,
      amenities: checkedAmenities
    };

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`Prediction request failed: ${res.status}`);
      const data = await res.json();

      if (data.success && data.predicted_price) {
        // Update price ranges and sqft metrics
        if (priceRangeOutput) {
          priceRangeOutput.textContent = data.price_range ? data.price_range.formatted : `₹${formatIndianNum(data.predicted_price * 0.9)} - ₹${formatIndianNum(data.predicted_price * 1.1)}`;
        }
        if (sqftOutput) {
          sqftOutput.textContent = data.formatted_sqft || `₹${formatIndianNum(data.price_per_sqft || 0)}/sq.ft`;
        }
        if (priceShortBadge) {
          priceShortBadge.textContent = data.short_price || `₹${(data.predicted_price / 100000).toFixed(2)} Lakhs`;
        }

        // Animate primary price counter
        animatePriceNumber(currentDisplayedPrice, data.predicted_price);
        currentDisplayedPrice = data.predicted_price;
      }
    } catch (err) {
      console.error('Error in fetchPrediction:', err);
    }
  }

  // 5. Dynamic Cascading Geography Handlers
  function populateCities(selectedState, targetCity = null) {
    if (!window.GEO_DATA || !citySelect) return;
    const citiesObj = window.GEO_DATA[selectedState] || {};
    const cities = Object.keys(citiesObj);

    citySelect.innerHTML = '';
    cities.forEach(city => {
      const opt = document.createElement('option');
      opt.value = city;
      opt.textContent = city;
      citySelect.appendChild(opt);
    });

    if (targetCity && cities.includes(targetCity)) {
      citySelect.value = targetCity;
    } else if (cities.length > 0) {
      citySelect.value = cities[0];
    }

    populateLocalities(selectedState, citySelect.value);
  }

  function populateLocalities(selectedState, selectedCity, targetLocality = null) {
    if (!window.GEO_DATA || !localitySelect) return;
    const cityData = window.GEO_DATA[selectedState]?.[selectedCity];
    const places = cityData ? cityData.places : [];

    localitySelect.innerHTML = '';
    places.forEach(place => {
      const opt = document.createElement('option');
      opt.value = place;
      opt.textContent = place;
      localitySelect.appendChild(opt);
    });

    if (targetLocality && places.includes(targetLocality)) {
      localitySelect.value = targetLocality;
    } else if (places.length > 0) {
      localitySelect.value = places[0];
    }

    triggerDebouncedInference();
  }

  // Event bindings for geography dropdowns
  if (stateSelect && citySelect && localitySelect) {
    stateSelect.addEventListener('change', () => {
      populateCities(stateSelect.value);
    });

    citySelect.addEventListener('change', () => {
      populateLocalities(stateSelect.value, citySelect.value);
    });

    localitySelect.addEventListener('change', () => {
      triggerDebouncedInference();
    });
  }

  if (monthSelect) {
    monthSelect.addEventListener('change', () => {
      updatePropertyAgeDisplay();
      triggerDebouncedInference();
    });
  }

  if (propTypeSelect) propTypeSelect.addEventListener('change', triggerDebouncedInference);
  if (furnishingSelect) furnishingSelect.addEventListener('change', triggerDebouncedInference);

  // 6. Sliders and Display Bindings
  const sliders = [
    { id: 'area_sqft', displayId: 'areaVal', format: v => `${Number(v).toLocaleString()} sq.ft` },
    { id: 'bedrooms', displayId: 'bedVal', format: v => `${v}` },
    { id: 'bathrooms', displayId: 'bathVal', format: v => `${Number(v).toFixed(1)}` },
    { id: 'garage_spaces', displayId: 'garageVal', format: v => `${v} car${v == 1 ? '' : 's'}` },
    { id: 'year_built', displayId: 'yearVal', format: v => `${v}` },
    { id: 'distance_to_city_center_km', displayId: 'distVal', format: v => `${Number(v).toFixed(1)} km` },
    { id: 'school_rating', displayId: 'schoolVal', format: v => `${Number(v).toFixed(1)} / 10` },
    { id: 'safety_index', displayId: 'safetyVal', format: v => `${v} / 100` },
  ];

  sliders.forEach(({ id, displayId, format }) => {
    const input = document.getElementById(id);
    const display = document.getElementById(displayId);
    if (input && display) {
      input.addEventListener('input', () => {
        display.textContent = format(input.value);
        if (id === 'year_built') {
          updatePropertyAgeDisplay();
        }
        triggerDebouncedInference();
      });
    }
  });

  // 7. Amenities Checkboxes
  const checkboxes = form ? form.querySelectorAll('input[type="checkbox"]') : [];
  checkboxes.forEach(cb => {
    cb.addEventListener('change', () => {
      updateAmenitiesCount();
      triggerDebouncedInference();
    });
  });

  function updateAmenitiesCount() {
    if (!form || !amenitiesCountOutput) return;
    const checked = form.querySelectorAll('input[name="amenity"]:checked');
    amenitiesCountOutput.textContent = `${checked.length} item${checked.length === 1 ? '' : 's'}`;
  }

  // 8. Tab Navigation
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(targetId);
      if (target) target.classList.add('active');
    });
  });

  // 9. Reset Button
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      if (stateSelect) stateSelect.value = 'Telangana';
      populateCities('Telangana', 'Hyderabad');
      populateLocalities('Telangana', 'Hyderabad', 'HITEC City & Cyber Towers');

      if (monthSelect) monthSelect.value = 'October';
      if (propTypeSelect) propTypeSelect.value = 'Apartment / Gated High-Rise Flat';
      if (furnishingSelect) furnishingSelect.value = 'Semi-Furnished';

      const defaults = {
        area_sqft: 2200,
        bedrooms: 3,
        bathrooms: 2.5,
        garage_spaces: 2,
        year_built: 2024,
        distance_to_city_center_km: 8.5,
        school_rating: 8.0,
        safety_index: 82
      };

      Object.entries(defaults).forEach(([key, val]) => {
        const el = document.getElementById(key);
        if (el) el.value = val;
      });

      sliders.forEach(({ id, displayId, format }) => {
        const input = document.getElementById(id);
        const display = document.getElementById(displayId);
        if (input && display) display.textContent = format(input.value);
      });

      updatePropertyAgeDisplay();

      checkboxes.forEach((cb, idx) => {
        cb.checked = [0, 2, 3].includes(idx);
      });

      updateAmenitiesCount();
      triggerDebouncedInference();
    });
  }

  // 10. Initial Page Boot
  if (stateSelect && window.GEO_DATA) {
    populateCities(stateSelect.value || 'Telangana', 'Hyderabad');
    populateLocalities(stateSelect.value || 'Telangana', 'Hyderabad', 'HITEC City & Cyber Towers');
  }

  updateAmenitiesCount();
  updatePropertyAgeDisplay();
  triggerDebouncedInference();
});
