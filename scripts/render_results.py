import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

out=Path('results'); out.mkdir(exist_ok=True)
eval=json.loads(Path('artifacts/evaluation.json').read_text())
rows=[]
for condition,models in [('clean',eval['clean'])]+[(f'{s} dB',m) for s,m in eval['noise'].items()]:
    for model,m in models.items():
        rows.append({'condition':condition,'model':model,'macro_f1':m['macro_f1'],'balanced_accuracy':m['balanced_accuracy']})
        if condition=='clean':
            fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(m['confusion_matrix'],cmap='Blues'); fig.colorbar(im,ax=ax)
            ax.set(xticks=range(8),yticks=range(8),xticklabels=m['class_order'],yticklabels=m['class_order'],xlabel='Predicted',ylabel='True',title=model)
            plt.setp(ax.get_xticklabels(),rotation=45,ha='right'); fig.tight_layout(); fig.savefig(out/f'{model}-confusion.png'); plt.close(fig)
            pd.DataFrame(m['classes']).T.to_csv(out/f'{model}-per-class.csv')
pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
Path('results/evaluation.json').write_text(json.dumps(eval,indent=2))
