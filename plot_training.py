"""
plot_training.py — Skipper NDT
================================
Affiche les courbes d'entraînement en temps réel pendant le training
ET génère un rapport final des courbes sauvegardé en PNG.

Courbes affichées :
  - Loss totale (train)
  - T1 : Accuracy + Recall (train vs val)
  - T2 : MAE en mètres (train vs val)

UTILISATION :
  Importez et appelez update_plots() à chaque époque dans model.py
  Les courbes se mettent à jour en temps réel dans une fenêtre matplotlib.

INTÉGRATION dans model.py :
  from plot_training import TrainingPlotter
  plotter = TrainingPlotter(n_epochs=100, output_dir='./models/')
  # dans la boucle d'entraînement, après evaluate() :
  plotter.update(epoch, avg_losses, train_metrics, val_metrics)
  # à la fin :
  plotter.save()
"""

import os
import matplotlib
matplotlib.use('TkAgg')   # affichage fenêtre — remplacer par 'Agg' si pas d'écran
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class TrainingPlotter:
    """
    Affiche et met à jour les courbes d'entraînement en temps réel.

    Args:
        n_epochs   : nombre total d'époques (pour l'axe X)
        output_dir : dossier où sauvegarder le PNG final
        live       : True = fenêtre en temps réel, False = sauvegarde uniquement
    """

    def __init__(self, n_epochs: int = 100,
                 output_dir: str = './models/',
                 live: bool = True):

        self.n_epochs   = n_epochs
        self.output_dir = output_dir
        self.live       = live

        # Historique
        self.epochs       = []
        self.loss_total   = []
        self.loss_t1      = []
        self.loss_t2      = []

        self.train_acc    = []
        self.train_recall = []
        self.train_f1     = []
        self.train_mae    = []

        self.val_acc      = []
        self.val_recall   = []
        self.val_f1       = []
        self.val_mae      = []

        # Objectifs du cahier des charges
        self.TARGET_ACC    = 0.92
        self.TARGET_RECALL = 0.95
        self.TARGET_MAE    = 1.0

        if self.live:
            self._init_figure()

    def _init_figure(self):
        """Initialise la fenêtre matplotlib avec 4 sous-graphiques."""
        plt.ion()   # mode interactif — mise à jour en temps réel
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('Skipper NDT — Courbes d\'entraînement',
                          fontsize=14, fontweight='bold')

        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.4, wspace=0.35)

        self.ax_loss    = self.fig.add_subplot(gs[0, 0])
        self.ax_acc     = self.fig.add_subplot(gs[0, 1])
        self.ax_recall  = self.fig.add_subplot(gs[1, 0])
        self.ax_mae     = self.fig.add_subplot(gs[1, 1])

        self._style_axes()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.1)

    def _style_axes(self):
        """Applique les titres et labels aux axes."""
        self.ax_loss.set_title('Loss totale (train)',   fontweight='bold')
        self.ax_loss.set_xlabel('Époque')
        self.ax_loss.set_ylabel('Loss')

        self.ax_acc.set_title('T1 — Accuracy',          fontweight='bold')
        self.ax_acc.set_xlabel('Époque')
        self.ax_acc.set_ylabel('Accuracy')
        self.ax_acc.set_ylim(0, 1.05)

        self.ax_recall.set_title('T1 — Recall',         fontweight='bold')
        self.ax_recall.set_xlabel('Époque')
        self.ax_recall.set_ylabel('Recall')
        self.ax_recall.set_ylim(0, 1.05)

        self.ax_mae.set_title('T2 — MAE (mètres)',      fontweight='bold')
        self.ax_mae.set_xlabel('Époque')
        self.ax_mae.set_ylabel('MAE (m)')

    def update(self, epoch: int,
               avg_losses: dict,
               val_metrics: dict,
               train_metrics: dict = None):
        """
        Met à jour l'historique et redessine les courbes.

        Args:
            epoch        : numéro de l'époque courante
            avg_losses   : {'loss_total': x, 'loss_t1': x, 'loss_t2': x}
            val_metrics  : {'accuracy': x, 'recall': x, 'f1': x, 'mae': x}
            train_metrics: idem (optionnel — calculé sur un sous-ensemble train)
        """
        self.epochs.append(epoch)

        # Losses
        self.loss_total.append(avg_losses.get('loss_total', 0))
        self.loss_t1.append(avg_losses.get('loss_t1', 0))
        self.loss_t2.append(avg_losses.get('loss_t2', 0))

        # Métriques val
        self.val_acc.append(val_metrics.get('accuracy', 0))
        self.val_recall.append(val_metrics.get('recall', 0))
        self.val_f1.append(val_metrics.get('f1', 0))
        mae = val_metrics.get('mae', float('nan'))
        self.val_mae.append(mae if not np.isnan(mae) else None)

        # Métriques train (optionnel)
        if train_metrics:
            self.train_acc.append(train_metrics.get('accuracy', 0))
            self.train_recall.append(train_metrics.get('recall', 0))
            self.train_f1.append(train_metrics.get('f1', 0))
            mae_tr = train_metrics.get('mae', float('nan'))
            self.train_mae.append(mae_tr if not np.isnan(mae_tr) else None)

        if self.live:
            self._redraw()

    def _redraw(self):
        """Efface et redessine tous les graphiques."""
        ep = self.epochs

        # ── Loss ──────────────────────────────────────────────────────────────
        self.ax_loss.cla()
        self.ax_loss.set_title('Loss (train)', fontweight='bold')
        self.ax_loss.set_xlabel('Époque')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.plot(ep, self.loss_total, 'b-',  lw=2,   label='Total')
        self.ax_loss.plot(ep, self.loss_t1,    'g--', lw=1.5, label='T1')
        self.ax_loss.plot(ep, self.loss_t2,    'r--', lw=1.5, label='T2')
        self.ax_loss.legend(fontsize=9)
        self.ax_loss.grid(True, alpha=0.3)

        # ── Accuracy (Train + Val sur le même graphique) ────────────────────────
        self.ax_acc.cla()
        self.ax_acc.set_title('T1 — Accuracy  (Train vs Val)', fontweight='bold')
        self.ax_acc.set_xlabel('Époque')
        self.ax_acc.set_ylabel('Accuracy')
        self.ax_acc.set_ylim(0, 1.05)
        # Val
        self.ax_acc.plot(ep, self.val_acc, color='steelblue', lw=2,
                         marker='o', markersize=3, label='Val')
        # Train
        if self.train_acc:
            self.ax_acc.plot(ep, self.train_acc, color='orange', lw=2,
                             marker='s', markersize=3, label='Train')
        # Ligne objectif
        self.ax_acc.axhline(self.TARGET_ACC, color='green', ls='--',
                            lw=1.5, label=f'Objectif {self.TARGET_ACC}')
        # Annotation meilleure val accuracy
        if self.val_acc:
            best_val = max(self.val_acc)
            best_ep  = ep[self.val_acc.index(best_val)]
            self.ax_acc.annotate(f'{best_val:.3f}',
                                 xy=(best_ep, best_val),
                                 xytext=(5, -12), textcoords='offset points',
                                 fontsize=8, color='steelblue')
        self.ax_acc.legend(fontsize=9, loc='lower right')
        self.ax_acc.grid(True, alpha=0.3)

        # ── Recall ────────────────────────────────────────────────────────────
        self.ax_recall.cla()
        self.ax_recall.set_title('T1 — Recall', fontweight='bold')
        self.ax_recall.set_xlabel('Époque')
        self.ax_recall.set_ylabel('Recall')
        self.ax_recall.set_ylim(0, 1.05)
        self.ax_recall.plot(ep, self.val_recall, 'r-', lw=2, label='Val')
        if self.train_recall:
            self.ax_recall.plot(ep, self.train_recall, 'r--', lw=1.5,
                                alpha=0.6, label='Train')
        self.ax_recall.axhline(self.TARGET_RECALL, color='green', ls=':',
                               lw=1.5, label=f'Objectif {self.TARGET_RECALL}')
        self.ax_recall.legend(fontsize=9)
        self.ax_recall.grid(True, alpha=0.3)

        # ── MAE ───────────────────────────────────────────────────────────────
        self.ax_mae.cla()
        self.ax_mae.set_title('T2 — MAE (mètres)', fontweight='bold')
        self.ax_mae.set_xlabel('Époque')
        self.ax_mae.set_ylabel('MAE (m)')

        mae_val_clean   = [v for v in self.val_mae if v is not None]
        ep_mae_val      = [ep[i] for i, v in enumerate(self.val_mae) if v is not None]
        if mae_val_clean:
            self.ax_mae.plot(ep_mae_val, mae_val_clean, 'g-', lw=2, label='Val')

        if self.train_mae:
            mae_tr_clean = [v for v in self.train_mae if v is not None]
            ep_mae_tr    = [ep[i] for i, v in enumerate(self.train_mae) if v is not None]
            if mae_tr_clean:
                self.ax_mae.plot(ep_mae_tr, mae_tr_clean, 'g--', lw=1.5,
                                 alpha=0.6, label='Train')

        self.ax_mae.axhline(self.TARGET_MAE, color='green', ls=':',
                            lw=1.5, label=f'Objectif {self.TARGET_MAE}m')
        self.ax_mae.legend(fontsize=9)
        self.ax_mae.grid(True, alpha=0.3)

        # Rafraîchissement
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def save(self, filename: str = 'training_curves.png'):
        """
        Sauvegarde les courbes finales en PNG haute résolution.
        Appelé automatiquement à la fin de l'entraînement.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)

        if not self.live:
            # Créer la figure pour la sauvegarde uniquement
            self._init_figure()
            if self.epochs:
                self._redraw()

        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Courbes sauvegardées → {save_path}")

        if self.live:
            plt.ioff()
            plt.show(block=False)

    def close(self):
        """Ferme la fenêtre matplotlib."""
        if self.live:
            plt.close(self.fig)