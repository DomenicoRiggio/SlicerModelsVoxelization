import qt

class HelpDialog(qt.QDialog):
    def __init__(self, parent=None):
        super(HelpDialog, self).__init__(parent)
        from VoxelizationLib.UI.utils import HELP_TEXT, CONTRIBUTORS
        self.setWindowTitle("Voxelization - Help Guide and Acknowledgements")
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)

        mainLayout = qt.QVBoxLayout(self)

        # Tab widget
        tabWidget = qt.QTabWidget()
        mainLayout.addWidget(tabWidget)

        # ── Help Guide tab ────────────────────────────────────────────────
        helpTab    = qt.QWidget()
        helpLayout = qt.QVBoxLayout(helpTab)
        helpBrowser = qt.QTextBrowser()
        helpBrowser.setOpenExternalLinks(True)
        helpBrowser.setReadOnly(True)
        helpBrowser.setHtml(HELP_TEXT)
        helpLayout.addWidget(helpBrowser)
        tabWidget.addTab(helpTab, "Help Guide")

        # ── Acknowledgements tab ──────────────────────────────────────────
        ackTab    = qt.QWidget()
        ackLayout = qt.QVBoxLayout(ackTab)
        ackBrowser = qt.QTextBrowser()
        ackBrowser.setReadOnly(True)

        contributors_html = "<h3>Contributors</h3><ul>"
        for contributor in CONTRIBUTORS:
            contributors_html += f"<li>{contributor}</li>"
        contributors_html += "</ul>"

        ackBrowser.setHtml(contributors_html)
        ackLayout.addWidget(ackBrowser)
        tabWidget.addTab(ackTab, "Acknowledgements")
