"""Reuse the v5 recipe with corrected training labels; keep validation/test fixed."""
import traceback
import train_expansion_v5 as recipe

recipe.WORK=recipe.ROOT/'expansion_v6'
recipe.DATA=recipe.WORK/'yolo26_paper1000_negative400'
recipe.BASE=recipe.ROOT/'training_runs/yolo26m_paper1000_negative400_v5/weights/best.pt'
recipe.NAME='yolo26m_paper1000_labelcleanup_v6'
recipe.RUN=recipe.ROOT/'training_runs'/recipe.NAME
recipe.STATE=recipe.WORK/'training_progress.json'

if __name__=='__main__':
    try:
        recipe.main()
    except Exception as exc:
        recipe.state('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
