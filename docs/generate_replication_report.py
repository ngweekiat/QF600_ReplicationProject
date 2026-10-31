"""Build the requested Word report from the current replication specification."""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from jkmp.config import load_config

cfg = load_config()
doc = Document()
sec = doc.sections[0]
sec.top_margin = sec.bottom_margin = Inches(0.75)
sec.left_margin = sec.right_margin = Inches(0.8)
normal = doc.styles['Normal']
normal.font.name = 'Calibri'
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(7)
normal.paragraph_format.line_spacing = 1.1
for name in ('Heading 1', 'Heading 2'):
    doc.styles[name].font.color.rgb = RGBColor.from_string('17365D')
    doc.styles[name].paragraph_format.keep_with_next = True

doc.add_heading('Machine Learning and the Implementable Efficient Frontier', 0)
doc.add_paragraph('Python replication: Static-ML* versus Portfolio-ML', 'Subtitle')
doc.add_paragraph('QF600 Asset Pricing | Specification as at 11 October 2026')
doc.add_paragraph('Research design and methodology draft. Empirical portfolio results are pending the WRDS data download.')

def p(text):
    doc.add_paragraph(text)

def equation(text):
    para = doc.add_paragraph(text)
    para.paragraph_format.left_indent = Inches(0.2)
    for run in para.runs:
        run.font.name = 'Cambria'
        run.font.size = Pt(11)

def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Light Shading Accent 1'
    for cell, value in zip(t.rows[0].cells, headers):
        cell.text = str(value)
    for row in rows:
        for cell, value in zip(t.add_row().cells, row):
            cell.text = str(value)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
    return t

doc.add_heading('1. Executive Summary', 1)
p('This project implements a modified Python replication of Jensen, Kelly, Malamud and Pedersen’s Machine Learning and the Implementable Efficient Frontier. Its central comparison is Static-ML*, which predicts stock returns before choosing a cost-aware portfolio, against Portfolio-ML, which learns portfolio coefficients using returns, risk and trading costs jointly. Markowitz-ML and the other benchmark portfolios have been excluded from the active comparison.')
p('The current configuration targets approximately 80 large US stocks listed on NYSE and Nasdaq and uses 50 characteristics across 13 JKP themes. The baseline investor has relative risk aversion of 10 and wealth of $10 billion at the end of 2025. Test-period returns cover January 2020 to December 2025. The implementation uses monthly portfolio decisions, a factor-based covariance model and quadratic market-impact costs.')
p('The research hypothesis is that directly learning an investable portfolio can improve net investor utility relative to a two-stage procedure. This is a hypothesis to test, rather than a result already established for this replication. The licensed data pull and empirical comparison remain incomplete; therefore, no performance numbers or claims of superiority are reported here. Numerical outcomes from the original paper cannot be assumed to carry over to this smaller universe and modified sample.')

doc.add_heading('2. Introduction: The Research Problem and Why It Matters', 1)
p('The research problem is an objective mismatch. A return-prediction model is generally selected for its ability to forecast individual stock returns, while an investor values the performance of a portfolio after risk and trading costs. A statistically useful signal may require frequent or expensive trades. Conversely, a persistent signal with modest forecasting accuracy may contribute more to a scalable investment strategy. The replication asks whether integrating implementation considerations into learning produces a better investment decision.')
p('The comparison is deliberately focused. Static-ML* is a demanding comparator because it already incorporates trading costs in its weight calculation and tunes its economic adjustments annually. Portfolio-ML must therefore demonstrate an advantage beyond simply subtracting costs from an otherwise cost-blind strategy. Both methods use the same investable universe, characteristic set, covariance estimates, investor settings and accounting rules.')
table(['Design element', 'Current specification'], [
    ('Investable universe', 'Target 80 eligible US stocks on NYSE and Nasdaq; selected by market capitalisation'),
    ('Membership rule', 'December reconstitution; new-member entry rank ≤70, incumbent retention rank ≤100; fill remaining slots to target'),
    ('Eligibility', 'Share codes 10, 11, 12; JKP common/primary_sec/obs_main flags; positive size and dollar volume; history and feature-coverage screens'),
    ('Characteristics', '50 inputs across 13 JKP themes; at least 25 available before imputation'),
    ('Periods', 'Data warm-up from 1993; configured training block 2005–2014; initial validation block 2015–2019; test returns 2020–2025'),
    ('Covariance estimation', 'Broader top-500 universe; 6 sector groups plus 13 characteristic themes; full 2,520-day window'),
    ('Investor baseline', 'γ=10; terminal wealth $10B at 31 December 2025'),
])
p('The 80-stock target is not a strict monthly count: retained members are not trimmed solely to enforce 80, and stocks can become ineligible during the year. Actual constituent names and realised counts require the data preparation stage. The return-prediction model is fitted on screened stocks with sufficient history, rather than only on the final portfolio members. Annual expanding estimation and selection rules govern the fitted models; the labelled date blocks do not describe a single frozen train/test exercise.')
p('The 50 characteristics were chosen in advance for coverage of valuation, quality, investment, price behaviour, risk and financial structure, rather than by ranking test-period returns. Their theme counts are value 6, momentum 5, quality 6, investment 5, low risk 5, short-term reversal 3, profitability 4, profit growth 4, low leverage 3, accruals 3, size 3, debt issuance 2 and seasonality 1. Theme assignments follow JKP metadata. Cross-sectional percentile ranks standardise the inputs; missing ranks are assigned 0.5.')

doc.add_heading('3. Theoretical Framework', 1)
doc.add_heading('2.1 From the Efficient Frontier to the Implementable Efficient Frontier', 2)
p('Jensen et al. (2026) evaluate investment opportunities using returns net of trading costs at each risk level. Their framework combines portfolio selection with machine learning and implementation costs [1]. This report applies that concept through the local code: an opportunity is assessed by what the investor can earn after executing the required trades.')
p('The frontier script reruns and retunes Static-ML* and Portfolio-ML over risk-aversion values {1, 5, 10, 20, 100} and investor wealth levels {$1B, $10B, $50B, $100B, $250B}. Each point therefore represents a portfolio optimised for its own investor settings. The figures compare annualised net excess return with annualised volatility; a capacity chart compares realised utility across wealth levels. The active plots contain the two selected methods and do not require a Markowitz-ML reference line.')

doc.add_heading('2.2 Trading Costs and Investor Size', 2)
p('The implementation models price impact as a quadratic function of changes in portfolio weights. Let wₜ be the chosen weights, wₜˢᵗᵃʳᵗ the inherited weights after return drift, Wₜ investor wealth, and Λₜ a diagonal matrix of stock-level impact coefficients. The fraction of wealth consumed by trading is:')
equation('TCₜ = ½ Wₜ (wₜ − wₜˢᵗᵃʳᵗ)ᵀ Λₜ (wₜ − wₜˢᵗᵃʳᵗ).')
equation('Λᵢ,ₜ = 2 × price_impact / dollar_volumeᵢ,ₜ = 0.2 / dollar_volumeᵢ,ₜ.')
p('Dollar volume uses the configured 126-day measure and is currently assumed to be denominated in dollars; that assumption must be verified against the actual WRDS data. Higher wealth increases the cost of a given weight change as a fraction of wealth. Lower liquidity also raises the penalty. This gives investor size an explicit role in portfolio construction instead of treating costs as a fixed deduction applied afterwards.')
p('The baseline wealth path is rolled backwards from end-2025 wealth using market total returns. It is a common externally specified size path, not the cumulative realised wealth of each strategy. The accounting charges sales of stocks leaving the universe when the required trading data remain available. The cost model represents market impact; separate borrowing fees, financing spreads and all real-world execution frictions are not explicitly modelled.')

doc.add_heading('2.3 The Optimal Solution: Aiming in Front of the Target', 2)
p('Portfolio-ML implements a dynamic trading rule that combines the inherited portfolio with a learned aim portfolio. Its trading-speed matrix is calculated from covariance, liquidity, investor wealth, risk aversion, the risk-free rate and expected portfolio growth:')
equation('wₜ = mₜ wₜˢᵗᵃʳᵗ + (I − mₜ) aimₜ.')
p('Here mₜ is the code’s retention matrix, so I−mₜ determines adjustment towards the aim. It is a matrix rather than a single percentage applied equally to every stock. The code uses ten fixed-point iterations, with expected monthly portfolio growth set to 0.007. The aim is constructed from current and lagged signals and fitted portfolio coefficients. By accounting for persistence and implementation, the investor can adjust gradually towards an economically useful position rather than fully replacing the portfolio each month.')
p('The phrase “aiming in front of the target” describes anticipation when adjustment is costly. It does not mean observing future realised returns. This implementation combines historical characteristic signals through trading-speed and drift matrices; it does not train separate return forecasts for all future horizons. The return-model configuration includes only the one-month horizon.')

doc.add_heading('2.4 Portfolio-ML vs Predict-then-Optimise (Static Portfolio)', 2)
p('Static-ML* first predicts next-month excess returns from random Fourier transformations of the 50 ranked characteristics using ridge regression. Candidate return models are selected by validation mean-squared error, then refitted before producing the next year’s predictions. The prediction feature-count grid is {2, 4, 8, 16, 32, 64, 128, 256, 512}. The portfolio stage subsequently combines predicted returns with risk, trading costs and inherited holdings:')
equation('wₜ = [γΣₜ + g diag(γΣₜ) + WₜkΛₜ]⁻¹ [u μ̂ₜ + WₜkΛₜ wₜˢᵗᵃʳᵗ].')
p('The code tunes return shrinkage u∈{0.25, 0.5, 1}, variance adjustment g∈{0, 1, 2}, and cost multiplier k∈{1, 1/3, 1/5}. These economic adjustments are selected each December using cumulative realised utility and applied to the next return year. “Static” identifies a one-period portfolio objective; the portfolio still rebalances monthly.')
p('Portfolio-ML instead estimates coefficients for transformed portfolio signals. Its reduced moments combine realised returns, covariance risk and the cost of changing the implied portfolio. It uses random-feature counts {64, 128, 256, 512}, with signals scaled by stock volatility. Its fitted coefficient vector takes the form:')
equation('β̂ = (Σ̄̃ + λI)⁻¹ r̄̃, where Σ̄̃ combines risk and trading-cost moments.')
p('Coefficients use prior realised-return data; feature settings and ridge penalties are chosen annually by cumulative realised utility. The key potential advantage is alignment: Portfolio-ML can prefer a signal because of its contribution to an implementable portfolio, even when it would not be preferred by a stock-level prediction criterion. However, Static-ML* also makes economic adjustments, and finite-sample estimation error can reverse the expected ranking.')
p('To show that Portfolio-ML is better in this replication, the empirical results must show higher out-of-sample net utility or a superior implementable frontier under matched assumptions. Higher gross return or lower prediction error alone is insufficient. At present, the code supplies the comparison design, not evidence of that ranking.')

doc.add_heading('2.5 Economic Feature Importance (Ridge Risk, Empirical Methods)', 2)
p('Ridge regularisation penalises large signal coefficients and improves stability when transformed predictors overlap or estimation is noisy. It is distinct from the covariance risk penalty: λ controls coefficient shrinkage, whereas γ controls investor risk aversion. The shared ridge grid contains zero plus 100 exponentially spaced penalties from exp(−10) to exp(10). In Portfolio-ML, the constant signal is penalised as well.')
p('Economic feature importance is measured by theme-level permutation, not by raw coefficient magnitude. For each theme, its characteristics are shuffled across the investable stocks within each month. The remaining characteristics are unchanged, and the same monthly permutation is reused across themes and methods. The code then recomputes portfolio outcomes and measures the decrease in annualised utility relative to the unshuffled portfolio.')
p('Portfolio-ML retains its fitted coefficients and hyperparameters while recomputing the shuffled signal-based aims. Static-ML* uses the fitted yearly return model to generate shuffled predictions, retains its selected annual economic adjustments, and reruns portfolio construction with trading costs. Both importance measures are evaluated net of costs. A positive utility drop indicates that disrupting the theme reduced performance; a negative drop indicates that the shuffled outcome performed better in that draw.')
p('These estimates describe model dependence in the observed sample. They do not establish causality, and correlated themes can share information. The baseline uses one monthly permutation generated from the configured seed. The 72-month test window and single permutation draw can make rankings unstable, so sensitivity to alternative seeds and specifications is a useful subsequent analysis. The script currently reports point estimates rather than confidence intervals.')

doc.add_heading('4. Performance and Evaluation Metrics', 1)
p('Monthly gross excess return is rₜ=Σᵢwᵢ,ₜrᵢ,ₜ₊₁ᵉˣᶜ; net excess return is rₜ−TCₜ. Holdings are formed at month-end t and evaluated over month t+1. Both methods start their test portfolios from value-weighted holdings, after which the previous holdings drift with returns before rebalancing. Performance statistics use sample variance with denominator n−1.')
table(['Metric', 'Definition in the current code', 'Interpretation'], [
    ('Gross excess return', '12 × mean(r)', 'Return before market-impact costs'),
    ('Net excess return', '12 × mean(r − TC)', 'Return after modelled trading costs'),
    ('Volatility', '√12 × sample SD(r)', 'Annualised gross-return variability'),
    ('Gross Sharpe ratio', '√12 × mean(r) / SD(r)', 'Gross excess return per unit of risk'),
    ('Net Sharpe ratio', '√12 × mean(r − TC) / SD(r)', 'Net mean divided by gross-return risk, following the implemented convention'),
    ('Annual trading costs', '12 × mean(TC)', 'Average yearly cost as a fraction of wealth'),
    ('Investor utility', '12 × [mean(r) − mean(TC) − ½γ Var(r)]', 'Primary risk- and cost-adjusted comparison'),
    ('Monthly turnover', 'Mean of Σᵢ|wᵢ − wᵢˢᵗᵃʳᵗ|', 'Sum of absolute traded weight changes; not halved'),
    ('Gross exposure', 'Mean of Σᵢ|wᵢ|', 'Average absolute risky-asset exposure'),
    ('Short exposure', 'Mean of Σᵢ|max(−wᵢ, 0)|', 'Average absolute short-position exposure'),
])
p('The net Sharpe denominator is intentionally the volatility of gross returns, not the volatility of the net-return series. Returns and costs are annualised arithmetically; the reported annual return is not a compounded CAGR. Utility is a realised mean–variance criterion and should not be described as a statistical significance test. Turnover, gross exposure and short exposure are reported as monthly averages.')
p('The base comparison is written to outputs/table2_test.csv. Supporting outputs contain chosen annual hyperparameters, monthly weights and portfolio returns. The frontier analysis explores risk aversion and investor size; the regime extension groups performance by recession status, volatility and interest-rate conditions and also re-costs trades using a VIX-based scaling. These supplementary checks help identify whether an advantage is broad or concentrated in particular conditions.')
p('A credible conclusion should compare the utility difference U(Portfolio-ML)−U(Static-ML*), net Sharpe ratios, frontier positions and trading costs, while checking whether leverage or short exposure explains the ranking. The existing scripts do not automatically establish statistical significance; paired uncertainty analysis would be an additional step. The smaller universe, modified characteristics, data conventions and pending volume-unit verification must be considered when interpreting the results.')
p('Empirical status: result cells should remain unfilled until the expanded 50-characteristic dataset, covariance estimates and models are rebuilt and the test-period portfolios are run. The implementation has passed 132 automated tests with three skips, but software validation does not substitute for empirical performance evidence.')

para = doc.add_paragraph()
para.add_run('References and implementation sources').bold = True
p('[1] Jensen, T. I., Kelly, B., Malamud, S., and Pedersen, L. H. (2026). Machine Learning and the Implementable Efficient Frontier. The Review of Financial Studies, 39(10), 3035–3078. https://doi.org/10.1093/rfs/hhag022. Used for the conceptual motivation; the detailed implementation descriptions above are based on the local Python code.')
p('[2] Original R replication repository: https://github.com/theisij/ml-and-the-implementable-efficient-frontier. Local copies are stored under reference/jkmp_r/. This project modifies the original design and is not a verbatim replication.')
p('[3] Local specification and code: configs/base.yaml; src/jkmp/models/return_model.py; src/jkmp/portfolio/static_ml.py, portfolio_ml.py, trading_speed.py and accounting.py; src/jkmp/evaluation/metrics.py and feature_importance.py; scripts/05_static_ml.py through scripts/10_regimes.py; docs/Selected comparison.md and docs/deviations.md.')

footer = sec.footer.paragraphs[0]
footer.alignment = 2
footer.add_run('QF600 replication | ')
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)
doc.core_properties.title = 'Static-ML* versus Portfolio-ML: Replication Framework'
doc.core_properties.subject = 'QF600 Asset Pricing replication methodology'
doc.core_properties.author = 'QF600 Replication Project'
output = Path(__file__).resolve().parent / 'QF600_Replication_Framework.docx'
doc.save(output)
print(output)
print('Characteristics:', len(cfg.features.names), '| Themes:', len(cfg.features.themes))
