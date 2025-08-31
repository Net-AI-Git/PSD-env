import matplotlib.pyplot as plt
import numpy as np
# נייבא את פונקציות העזר שלנו מהקובץ הקיים
from optimizer_core import psd_utils as utils


def main():
    """
    סקריפט זה מיועד לבדיקה ויזואלית בלבד.
    הוא טוען את נתוני ה-PSD, יוצר את מאגר נקודות המועמדים באותה שיטה
    כמו האלגוריתם הראשי, ומציג אותן על גבי הגרף המקורי.
    """
    # --- הגדרות זהות לאלגוריתם הראשי ---
    FILENAME = "A1X.txt"
    WINDOW_SIZES = [10, 20, 30]

    # --- שלב 1: טעינת הנתונים ויצירת מאגר המועמדים ---
    frequencies, psd_values = utils.read_psd_data(FILENAME)
    if frequencies is None:
        print("File not found or is empty.")
        return

    # זו אותה פונקציה בדיוק שהאלגוריתם הראשי משתמש בה
    candidate_points = utils.create_multi_scale_envelope(frequencies, psd_values, WINDOW_SIZES)

    print(f"\nFound a total of {len(candidate_points)} unique candidate points to visualize.")

    # --- שלב 2: יצירת התצוגה הוויזואלית ---
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle("Visualization of All Candidate Points", fontsize=18)

    for ax, x_scale in zip(axes, ["log", "linear"]):
        ax.set_title(f"Candidate Points Pool ({x_scale.capitalize()} X-axis)", fontsize=14)
        ax.set_xscale(x_scale)
        ax.set_yscale('log')

        # שרטוט אות ה-PSD המקורי בצבע כחול
        ax.plot(frequencies, psd_values, 'b-', label='Original PSD', alpha=0.7)

        # שרטוט כל נקודות המועמדים במאגר כנקודות אדומות
        ax.plot(candidate_points[:, 0], candidate_points[:, 1], 'ro',
                label=f'All Candidate Points ({len(candidate_points)})', markersize=4)

        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('PSD [g²/Hz]', fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
