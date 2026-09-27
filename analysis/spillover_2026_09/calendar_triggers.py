"""The 34 non-Kalshi calendar releases used as extra triggers: release name →
(family, hawkish sign). Shared by ``releases.py`` and ``augmented.py`` (kept separate so the
trigger list has one home)."""

# trigger → (family, hawkish sign). Headlines only; components of the same
# release are left out so one release is not counted several times.
TRIGGERS = {
    # inflation-type: higher = hawkish
    "Core PPI MoM": ("inflation", +1), "PPI Ex Food, Energy and Trade MoM": ("inflation", +1),
    "Import Prices MoM": ("inflation", +1), "ISM Manufacturing Prices": ("inflation", +1),
    "ISM Services Prices": ("inflation", +1), "ISM Non-Manufacturing Prices": ("inflation", +1),
    "Michigan Inflation Expectations": ("inflation", +1),
    "Michigan 5 Year Inflation Expectations": ("inflation", +1),
    "Consumer Inflation Expectation": ("inflation", +1), "Unit Labour Costs QoQ": ("inflation", +1),
    # activity: higher = hawkish (stronger economy)
    "Retail Sales MoM": ("activity", +1), "Retail Sales Ex Autos MoM": ("activity", +1),
    "ISM Manufacturing PMI": ("activity", +1), "ISM Services PMI": ("activity", +1),
    "ISM Non-Manufacturing PMI": ("activity", +1), "S&P Global Manufacturing PMI": ("activity", +1),
    "S&P Global Services PMI": ("activity", +1), "Durable Goods Orders MoM": ("activity", +1),
    "Industrial Production MoM": ("activity", +1), "Philadelphia Fed Manufacturing Index": ("activity", +1),
    "NY Empire State Manufacturing Index": ("activity", +1), "Chicago PMI": ("activity", +1),
    "Personal Spending MoM": ("activity", +1), "Personal Income MoM": ("activity", +1),
    "Factory Orders MoM": ("activity", +1), "Existing Home Sales": ("activity", +1),
    "Building Permits": ("activity", +1), "NAHB Housing Market Index": ("activity", +1),
    "Atlanta Fed GDPNow": ("activity", +1), "Construction Spending MoM": ("activity", +1),
    # labour
    "JOLTs Job Openings": ("labour", +1), "Continuing Jobless Claims": ("labour", -1),
    # sentiment
    "Michigan Consumer Sentiment": ("sentiment", +1), "CB Consumer Confidence": ("sentiment", +1),
}
