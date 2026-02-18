# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGraphicsView,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1103, 662)
        Widget.setMinimumSize(QSize(720, 400))
        Widget.setMaximumSize(QSize(1920, 1080))
        self.horizontalLayout_3 = QHBoxLayout(Widget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.gridWidget = QWidget(Widget)
        self.gridWidget.setObjectName(u"gridWidget")
        self.gridLayout = QGridLayout(self.gridWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.graphicsViewRes = QGraphicsView(self.gridWidget)
        self.graphicsViewRes.setObjectName(u"graphicsViewRes")

        self.gridLayout.addWidget(self.graphicsViewRes, 8, 1, 1, 1)

        self.line = QFrame(self.gridWidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line, 4, 0, 1, 2)

        self.graphicsView = QGraphicsView(self.gridWidget)
        self.graphicsView.setObjectName(u"graphicsView")

        self.gridLayout.addWidget(self.graphicsView, 8, 0, 1, 1)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.gridWidget)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout.addWidget(self.label)

        self.comboBoxModel = QComboBox(self.gridWidget)
        self.comboBoxModel.setObjectName(u"comboBoxModel")

        self.horizontalLayout.addWidget(self.comboBoxModel)

        self.label_4 = QLabel(self.gridWidget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMaximumSize(QSize(100, 16777215))

        self.horizontalLayout.addWidget(self.label_4)

        self.comboBoxTask = QComboBox(self.gridWidget)
        self.comboBoxTask.setObjectName(u"comboBoxTask")
        self.comboBoxTask.setMinimumSize(QSize(100, 0))

        self.horizontalLayout.addWidget(self.comboBoxTask)

        self.label_5 = QLabel(self.gridWidget)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout.addWidget(self.label_5)

        self.comboBoxDevice = QComboBox(self.gridWidget)
        self.comboBoxDevice.setObjectName(u"comboBoxDevice")

        self.horizontalLayout.addWidget(self.comboBoxDevice)

        self.btnLoadModel = QPushButton(self.gridWidget)
        self.btnLoadModel.setObjectName(u"btnLoadModel")
        self.btnLoadModel.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout.addWidget(self.btnLoadModel)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 2)

        self.labelModelInfo = QLabel(self.gridWidget)
        self.labelModelInfo.setObjectName(u"labelModelInfo")

        self.gridLayout.addWidget(self.labelModelInfo, 3, 0, 1, 2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btnOpenImage = QPushButton(self.gridWidget)
        self.btnOpenImage.setObjectName(u"btnOpenImage")
        self.btnOpenImage.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_2.addWidget(self.btnOpenImage)

        self.btnRunInference = QPushButton(self.gridWidget)
        self.btnRunInference.setObjectName(u"btnRunInference")
        self.btnRunInference.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        self.horizontalLayout_2.addWidget(self.btnRunInference)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)


        self.gridLayout.addLayout(self.horizontalLayout_2, 6, 0, 1, 2)

        self.labelProcess = QLabel(self.gridWidget)
        self.labelProcess.setObjectName(u"labelProcess")
        self.labelProcess.setLineWidth(1)

        self.gridLayout.addWidget(self.labelProcess, 10, 0, 1, 2)

        self.line_2 = QFrame(self.gridWidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)

        self.gridLayout.addWidget(self.line_2, 11, 0, 1, 2)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalSliderNMS = QSlider(self.gridWidget)
        self.horizontalSliderNMS.setObjectName(u"horizontalSliderNMS")
        self.horizontalSliderNMS.setMaximum(100)
        self.horizontalSliderNMS.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalSliderNMS.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout_3.addWidget(self.horizontalSliderNMS, 2, 2, 1, 1)

        self.lineEditConf = QLineEdit(self.gridWidget)
        self.lineEditConf.setObjectName(u"lineEditConf")
        self.lineEditConf.setMaximumSize(QSize(100, 16777215))

        self.gridLayout_3.addWidget(self.lineEditConf, 1, 1, 1, 1)

        self.horizontalSliderConf = QSlider(self.gridWidget)
        self.horizontalSliderConf.setObjectName(u"horizontalSliderConf")
        self.horizontalSliderConf.setMaximum(100)
        self.horizontalSliderConf.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalSliderConf.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.gridLayout_3.addWidget(self.horizontalSliderConf, 1, 2, 1, 1)

        self.lineEditNMS = QLineEdit(self.gridWidget)
        self.lineEditNMS.setObjectName(u"lineEditNMS")
        self.lineEditNMS.setMaximumSize(QSize(100, 16777215))

        self.gridLayout_3.addWidget(self.lineEditNMS, 2, 1, 1, 1)

        self.label_2 = QLabel(self.gridWidget)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)

        self.labelNMS = QLabel(self.gridWidget)
        self.labelNMS.setObjectName(u"labelNMS")
        self.labelNMS.setMaximumSize(QSize(5000, 16777215))

        self.gridLayout_3.addWidget(self.labelNMS, 2, 0, 1, 1)


        self.gridLayout.addLayout(self.gridLayout_3, 12, 0, 1, 2)


        self.horizontalLayout_3.addWidget(self.gridWidget)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.label.setText(QCoreApplication.translate("Widget", u"Model: ", None))
        self.label_4.setText(QCoreApplication.translate("Widget", u"Task:", None))
        self.label_5.setText(QCoreApplication.translate("Widget", u"Device: ", None))
        self.btnLoadModel.setText(QCoreApplication.translate("Widget", u"Load Model", None))
        self.labelModelInfo.setText("")
        self.btnOpenImage.setText(QCoreApplication.translate("Widget", u"Open Image", None))
        self.btnRunInference.setText(QCoreApplication.translate("Widget", u"Inference", None))
        self.labelProcess.setText(QCoreApplication.translate("Widget", u"Process Time:", None))
        self.label_2.setText(QCoreApplication.translate("Widget", u"Confidence Threshold:", None))
        self.labelNMS.setText(QCoreApplication.translate("Widget", u"NMS Threshold:", None))
    # retranslateUi

