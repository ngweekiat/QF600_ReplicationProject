"""Frozen transparent group/quota/redundancy rules; no ML fit."""
import numpy as np
import pandas as pd

def assign_themes(registry,corr):
    result=registry.copy();groups=result.loc[result.published_theme.notna()].groupby('published_theme').characteristic.apply(list).to_dict()
    themes=[];methods=[];strengths=[]
    for row in result.itertuples():
        if pd.notna(row.published_theme):themes.append(row.published_theme);methods.append('published JKP cluster');strengths.append(np.nan)
        else:
            proxy={theme:float(corr.loc[row.characteristic,chars].abs().mean()) for theme,chars in groups.items()}
            theme=max(proxy,key=lambda k:proxy[k] if np.isfinite(proxy[k]) else -np.inf)
            themes.append(theme);methods.append('empirical training correlation proxy; not published semantic cluster');strengths.append(proxy[theme])
    result['theme']=themes;result['theme_assignment_method']=methods;result['theme_assignment_strength']=strengths
    return result

def diverse_pick(scores,corr,count,min_theme=2,max_theme=None,score_col='screen_score',redundancy_penalty=.15,published_anchor=False):
    """Greedy transparent coverage/diversity selection, not a global-optimality claim."""
    pool=scores.loc[scores.eligible].copy().set_index('characteristic',drop=False)
    if len(pool)<count:raise ValueError(f'Only {len(pool)} coverage-qualified candidates for requested {count}')
    chosen=[];reasons={};theme_counts={}
    def add(char,reason):
        chosen.append(char);theme=pool.loc[char,'theme'];theme_counts[theme]=theme_counts.get(theme,0)+1;reasons[char]=reason
    def best(chars):
        values={}
        for char in chars:
            redundant=float(corr.loc[char,chosen].abs().max()) if chosen else 0.
            if not np.isfinite(redundant):redundant=0.
            # Small documented preference for published interpretable cluster membership.
            values[char]=float(pool.loc[char,score_col])+.02*bool(pool.loc[char,'published_jkp_153'])-redundancy_penalty*redundant
        return max(values,key=lambda c:(values[c],c))
    for theme in sorted(pool.theme.unique()):
        chars=pool.index[pool.theme.eq(theme)].tolist()
        if published_anchor:
            anchors=[c for c in chars if pool.loc[c,'published_jkp_153']]
            if not anchors:raise ValueError(f'No eligible published representative for theme {theme}')
            if len(chosen)>=count:raise ValueError('Theme anchors make requested subset infeasible')
            add(best(anchors),'published JKP representative for economic theme')
        for _ in range(max(0,min(min_theme,len(chars))-theme_counts.get(theme,0))):
            available=[c for c in chars if c not in chosen]
            if not available or len(chosen)>=count:break
            add(best(available),'minimum economic-theme representation')
    while len(chosen)<count:
        available=[c for c in pool.index if c not in chosen and (max_theme is None or theme_counts.get(pool.loc[c,'theme'],0)<max_theme)]
        if not available:raise ValueError('Theme cap makes requested subset infeasible; no silent relaxation')
        add(best(available),'balanced predictive score with redundancy penalty')
    selected=pool.loc[chosen].copy();selected['selection_order']=np.arange(1,len(selected)+1);selected['selection_reason']=[reasons[c] for c in chosen]
    return selected.reset_index(drop=True)
