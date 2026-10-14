"""Build ~/dataset-page/site/index.html from the README that analysis.py just updated in ~/dataset-page/work: the
dataset sections (overview, description with its figures, detections, contributors), with the figures served from
here instead of GitHub. Run by build_dataset_page.sh after analysis.py; served on :8772 by ~/review_host.sh."""
import os
import re
import time
from pathlib import Path

import markdown

HOME = Path.home()
WORK, SITE = HOME / 'dataset-page' / 'work', HOME / 'dataset-page' / 'site'
SECTIONS = [('## Dataset overview', '## Citation and usage of code'),
            ('## Description and analysis of dataset', '## Adding videos to dataset'),
            ('## Who added the videos', '## Contact')]


def section(text, start, end):
    i = text.find(start)
    j = text.find(end, i + 1)
    return text[i:j] if i >= 0 and j > i else ''


def local_links(md):
    """Figures from here: the GitHub preview and blob links of figures/X become figures/X."""
    md = re.sub(r'https://htmlpreview\.github\.io/\?https://github\.com/crowd-dataset/crowd/blob/main/'
                r'(figures/[^)\s"\']+)',
                r'\1', md)
    return re.sub(r'https://(?:github\.com/crowd-dataset/crowd/(?:blob|raw)/main|'
                  r'raw\.githubusercontent\.com/crowd-dataset/'
                  r'crowd/main)/(figures/[^)\s"\']+)', r'\1', md)


def build():
    readme = (WORK / 'README.md').read_text()
    md = '\n\n'.join(filter(None, (section(readme, a, b) for a, b in SECTIONS)))
    # the README's note that figures open through htmlpreview.github.io does not apply: here they open from the mini
    md = local_links(md).replace('<details>', '<details markdown="1">')
    md = '\n\n'.join(p for p in md.split('\n\n') if 'htmlpreview.github.io' not in p)  # after local_links
    body = markdown.markdown(md, extensions=['tables', 'md_in_html', 'attr_list'])
    built = time.strftime('%Y-%m-%d %H:%M')
    SITE.mkdir(parents=True, exist_ok=True)
    if not (SITE / 'figures').exists():
        os.symlink(WORK / 'figures', SITE / 'figures')
    (SITE / 'index.html.tmp').write_text(PAGE.replace('__BUILT__', built).replace('__BODY__', body))
    os.replace(SITE / 'index.html.tmp', SITE / 'index.html')
    print(f'{built} page built: {len(body) // 1024} kB')


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CROWD dataset</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>
<text y=%22.9em%22 font-size=%2290%22>🚐</text></svg>">
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--muted:#667085;--line:#e4e7ec;--link:#2563eb;--head:#eef0f3}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8ec;--muted:#8b93a1;--line:#272c36;
 --link:#60a5fa;--head:#1d2129}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px 16px 48px}
header{padding:20px 16px 4px;max-width:1000px;margin:0 auto}
header h1{margin:0;font-size:24px}header p{margin:4px 0 0;color:var(--muted);font-size:13px}
a{color:var(--link)}h2{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:6px}h3{margin-top:26px}
img{max-width:100%;height:auto;border-radius:8px;background:#fff}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0;display:block;overflow-x:auto}
th,td{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:var(--head);font-weight:600}td:nth-child(n+3){font-variant-numeric:tabular-nums}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin:10px 0}
summary{cursor:pointer}code{font-size:90%;background:var(--head);padding:1px 4px;border-radius:4px}
</style></head><body>
<header><h1>🚐 CROWD dataset</h1>
<p>Built nightly on the mini from the review mapping, including reviews not yet pushed to GitHub ·
last built __BUILT__ ·
<a href="#" onclick="location.port=8770;return false">review hub</a> ·
<a href="#" onclick="location.port=8771;return false">pipeline</a> ·
<a href="#" onclick="location.port=8090;return false">machines</a></p></header>
<main>__BODY__</main></body></html>"""

if __name__ == '__main__':
    build()
