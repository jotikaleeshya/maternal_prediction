
const API_BASE_URL = "http://127.0.0.1:5000"; 

function goToHome() {
    window.location.href = "index.html";
}

function goToHistory() {
    window.location.href = "history.html";
}

function goToInput() {
    window.location.href = "input.html";
}

function goToResult() {
    window.location.href = "result.html";
}

async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: { "Content-Type": "application/json" },
            ...options,
        });

        if (!response.ok) {
            const errorData = await response.json();
            return errorData;
        }

        return await response.json();
    } catch (err) {
        console.error("API ERROR:", err);
        return { success: false, message: "Tidak dapat terhubung ke server. Pastikan server API berjalan di http://127.0.0.1:5000" };
    }
}

async function loadHomeData() {

    const userName = document.getElementById("userName");
    if (userName) {
        userName.innerText = "Selamat datang di Materlife!";
    }


    try {
        const result = await fetchAPI("/stats");

        if (result && result.success) {
            const data = result.data;

            document.getElementById("bloodPressure").innerText = data.avgBloodPressure || "--";
            document.getElementById("bloodSugar").innerText = data.avgBloodSugar || "--";
            document.getElementById("bodyTemp").innerText = data.avgBodyTemp || "--";
            document.getElementById("heartRate").innerText = data.avgHeartRate || "--";
        } else {

            document.getElementById("bloodPressure").innerText = "--";
            document.getElementById("bloodSugar").innerText = "--";
            document.getElementById("bodyTemp").innerText = "--";
            document.getElementById("heartRate").innerText = "--";
        }
    } catch (error) {
        console.error("Error loading home data:", error);
        document.getElementById("bloodPressure").innerText = "--";
        document.getElementById("bloodSugar").innerText = "--";
        document.getElementById("bodyTemp").innerText = "--";
        document.getElementById("heartRate").innerText = "--";
    }
}

async function submitHealthData(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    const submitButton = form.querySelector(".submit-button");
    const originalText = submitButton.textContent;
    submitButton.textContent = "Memproses...";
    submitButton.disabled = true;

    const data = {
        Age: Number(formData.get("age")),
        SystolicBP: Number(formData.get("sbp")),
        DiastolicBP: Number(formData.get("dbp")),
        BS: parseFloat(formData.get("bs")),
        BodyTemp: parseFloat(formData.get("temp")),
        HeartRate: Number(formData.get("hr"))
    };

    for (const key in data) {
        if (data[key] === null || data[key] === undefined || isNaN(data[key]) || data[key] === 0) {
            alert(`Field ${key} harus diisi dengan angka valid (tidak boleh kosong atau 0).`);
            submitButton.textContent = originalText;
            submitButton.disabled = false;
            return;
        }
    }

    console.log("Payload ke API:", data);

    try {
        const result = await fetchAPI("/predict", {
            method: "POST",
            body: JSON.stringify(data)
        });

        console.log("Full API response:", result);

        if (result && result.success) {
          
            sessionStorage.setItem("latestRisk", result.risk);
            sessionStorage.setItem("latestData", JSON.stringify(data));
            sessionStorage.setItem("latestWarnings", JSON.stringify(result.warnings || []));
            sessionStorage.setItem("latestRecommendations", JSON.stringify(result.recommendations || []));
            goToResult();
        } else {
            alert(result?.message || "Gagal memproses data. Pastikan input lengkap.");
        }
    } catch (error) {
        console.error("Error submitting health data:", error);
        alert("Terjadi kesalahan. Silakan coba lagi.");
    }

    submitButton.textContent = originalText;
    submitButton.disabled = false;
}

async function loadResultData() {
    const risk = sessionStorage.getItem("latestRisk");

    if (!risk) {
        document.getElementById("riskCategory").innerText = "Kategori: Tidak Ditemukan";
        document.getElementById("scoreNumber").innerText = "--";
        document.getElementById("scoreStatus").innerText = "Data tidak tersedia";
        return;
    }

    document.getElementById("riskCategory").innerText = `Kategori: ${risk.toUpperCase()}`;

    let score = 0;
    let statusText = "";
    let riskPercentage = 0;

    if (risk === "low risk") {
        score = 90;
        statusText = "Kondisi Baik";
        riskPercentage = 10;
    } else if (risk === "mid risk") {
        score = 60;
        statusText = "Perlu Perhatian";
        riskPercentage = 40;
    } else if (risk === "high risk") {
        score = 30;
        statusText = "Risiko Tinggi";
        riskPercentage = 70;
    }

    document.getElementById("scoreNumber").innerText = score;
    document.getElementById("scoreStatus").innerText = statusText;

    const riskNumber = document.getElementById("riskNumber");
    if (riskNumber) {
        riskNumber.innerText = riskPercentage;
    }

    const circle = document.getElementById("scoreProgress");
    if (circle) {
        const circumference = 2 * Math.PI * 88;
        const offset = circumference - (score / 100) * circumference;

        setTimeout(() => {
            circle.style.strokeDashoffset = offset;
        }, 100);
    }

    const riskCircle = document.getElementById("riskProgress");
    if (riskCircle) {
        const circumference = 2 * Math.PI * 36;
        const offset = circumference - (riskPercentage / 100) * circumference;

        setTimeout(() => {
            riskCircle.style.strokeDashoffset = offset;
        }, 100);
    }

    loadWarningsAndRecommendations();
}

function loadWarningsAndRecommendations() {
    const warningsData = JSON.parse(sessionStorage.getItem("latestWarnings") || "[]");
    const recommendationsData = JSON.parse(sessionStorage.getItem("latestRecommendations") || "[]");

    const warningsList = document.getElementById("warningsList");
    const recommendationsList = document.getElementById("recommendationsList");

    if (warningsList) {
        if (warningsData.length > 0) {
            warningsList.innerHTML = warningsData.map(warning =>
                `<div class="warning-item">${warning}</div>`
            ).join('');
        } else {
            warningsList.innerHTML = '<p class="no-warnings">Tidak ada peringatan khusus</p>';
        }
    }

    if (recommendationsList) {
        if (recommendationsData.length > 0) {
            recommendationsList.innerHTML = recommendationsData.map(rec =>
                `<div class="recommendation-item">${rec}</div>`
            ).join('');
        } else {
            recommendationsList.innerHTML = '<p class="no-recommendations">Tidak ada rekomendasi tambahan</p>';
        }
    }
}

async function loadHistoryData() {
    const historyList = document.getElementById("historyList");

    if (!historyList) return;

    historyList.innerHTML = '<div class="loading">Memuat riwayat...</div>';

    try {
        const result = await fetchAPI("/history");

        if (result && result.success && result.data.length > 0) {
            historyList.innerHTML = "";

            result.data.forEach((entry) => {
                const historyItem = document.createElement("div");
                historyItem.className = "history-item";

                const riskClass = entry.risk.toLowerCase().replace(" ", "-");
                const riskLabel = entry.risk.toUpperCase();

                historyItem.innerHTML = `
                    <div class="history-header">
                        <div class="history-date">${formatDate(entry.date)}</div>
                        <div class="history-risk ${riskClass}">${riskLabel}</div>
                    </div>
                    <div class="history-details">
                        <div class="history-detail-item">
                            <span class="detail-label">Usia:</span>
                            <span class="detail-value">${entry.age} tahun</span>
                        </div>
                        <div class="history-detail-item">
                            <span class="detail-label">Tekanan Darah:</span>
                            <span class="detail-value">${entry.systolicBP}/${entry.diastolicBP} mmHg</span>
                        </div>
                        <div class="history-detail-item">
                            <span class="detail-label">Gula Darah:</span>
                            <span class="detail-value">${entry.bs} mmol/L</span>
                        </div>
                        <div class="history-detail-item">
                            <span class="detail-label">Suhu Tubuh:</span>
                            <span class="detail-value">${entry.bodyTemp} °C</span>
                        </div>
                        <div class="history-detail-item">
                            <span class="detail-label">Detak Jantung:</span>
                            <span class="detail-value">${entry.heartRate} bpm</span>
                        </div>
                    </div>
                `;

                historyList.appendChild(historyItem);
            });
        } else {
            historyList.innerHTML = `
                <div class="empty-history">
                    <p>Belum ada riwayat tes kesehatan</p>
                    <button class="submit-button" onclick="goToInput()">Mulai Tes</button>
                </div>
            `;
        }
    } catch (error) {
        console.error("Error loading history:", error);
        historyList.innerHTML = `
            <div class="error-message">
                <p>Gagal memuat riwayat. Pastikan server API berjalan.</p>
                <button class="submit-button" onclick="loadHistoryData()">Coba Lagi</button>
            </div>
        `;
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const options = {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return date.toLocaleDateString('id-ID', options);
}

function clearInput(button) {
    const input = button.parentElement.querySelector(".input-field");
    input.value = "";
    input.focus();
    button.style.display = "none";
}

document.addEventListener("DOMContentLoaded", () => {
    document.documentElement.style.scrollBehavior = "smooth";

    const form = document.getElementById("healthCheckForm");
    if (form) {
        form.addEventListener("submit", submitHealthData);
    }

    const inputs = document.querySelectorAll(".input-field");
    inputs.forEach((input) => {
        input.addEventListener("input", function () {
            const clearBtn = this.parentElement.querySelector(".clear-button");
            if (clearBtn) clearBtn.style.display = this.value ? "block" : "none";
        });
    });

    if (document.getElementById("riskCategory")) {
        loadResultData();
    }

    if (document.getElementById("historyList")) {
        loadHistoryData();
    }

    if (document.getElementById("userName")) {
        loadHomeData();
    }
});
