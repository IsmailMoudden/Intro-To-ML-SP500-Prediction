# Technical Indicators

Technical indicators are essential tools in financial analysis, providing insights into market momentum, trend direction, and potential reversal points. Below are some of the most commonly used indicators:

---

## RSI (Relative Strength Index)

- **Description:**  
  A momentum oscillator that measures the speed and change of price movements.
  
- **Purpose:**  
  Helps determine overbought or oversold conditions in the market.

- **Typical Window:**  
  14 days.

- **Usage:**  
  When the RSI exceeds a certain level (commonly 70), it suggests that the asset may be overbought, whereas an RSI below 30 indicates that it may be oversold.

---

## MACD (Moving Average Convergence Divergence)

- **Description:**  
  A trend-following momentum indicator that reveals the relationship between two moving averages of a security’s price.

- **Purpose:**  
  Helps identify changes in the strength, direction, momentum, and duration of a trend.

- **Calculation:**  
  - **MACD Line:** Difference between a 12-day Exponential Moving Average (EMA) and a 26-day EMA.
  - **Signal Line:** A 9-day EMA of the MACD line.
  
- **Usage:**  
  Crossovers between the MACD line and the signal line can indicate potential buy or sell signals.

---

## SMA (Simple Moving Average)

- **Description:**  
  A straightforward indicator that calculates the average of a security’s closing prices over a specified number of days.

- **Purpose:**  
  Smooths out price fluctuations to make it easier to identify underlying trends.

- **Calculation:**  
  SMA = (Sum of Closing Prices over N Days) / N  
  *For example, a 20-day SMA averages the closing prices over the last 20 days.*

- **Common Uses:**  
  - **Short-term Trends:**  
    A 20-day SMA is typically used to gauge short-term price trends.
  - **Long-term Trends:**  
    Longer SMAs, such as the 50-day or 100-day SMA, help in identifying longer-term trends.

---

### Bollinger Bands

- **Description :**  
  Les Bollinger Bands consistent en une moyenne mobile (souvent un SMA sur 20 jours) entourée de deux bandes placées à un nombre défini d'écarts-types (généralement 2) de part et d'autre.  
- **But :**  
  Ils mesurent la volatilité du marché et peuvent indiquer des conditions de surachat ou de survente.  
- **Utilisation :**  
  Lorsque le cours touche la bande supérieure, cela peut indiquer un niveau de surachat; s’il touche la bande inférieure, il peut suggérer une situation de survente.

---

## Explications des Calculs Techniques

- **RSI (Relative Strength Index) :**  
  Le RSI est calculé en comparant la moyenne des gains et la moyenne des pertes sur une période donnée (souvent 14 jours). La formule standard est :  
  RSI = 100 - (100 / (1 + (avg_gain/avg_loss)))  
  Ici, "avg_gain" et "avg_loss" représentent respectivement la moyenne des gains et pertes sur la période. La soustraction du résultat à 100 permet d'obtenir un indice où 0 indique une forte pression baissière et 100 une forte pression haussière.

- **MACD (Moving Average Convergence Divergence) :**  
  Le MACD est obtenu en soustrayant la moyenne mobile exponentielle (EMA) à 26 jours de l'EMA à 12 jours. Ensuite, une ligne de signal est calculée en prenant l'EMA du MACD sur 9 jours. La « MACD Diff » (différence entre le MACD et sa ligne de signal) met en évidence l'intensité de la tendance.

- **Utilisation de -1 :**  
  L'usage de "-1" avec des méthodes comme iloc[-1] permet de sélectionner la dernière valorisation calculée. Cela garantit que, pour la prédiction, nous utilisons les indicateurs les plus récents (par exemple, le dernier SMA ou RSI).

---

By combining these indicators, analysts can gain a well-rounded view of market conditions, making it easier to spot potential trading opportunities or warning signs.
