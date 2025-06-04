import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QComboBox, QPushButton, QListWidget, 
                             QMessageBox, QHBoxLayout, QSlider, QListWidgetItem)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont

# ... (остальной код класса AdvancedBookRecommender остается без изменений)

class RecommenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Четкий книжный рекомендатель Artemka 3000")
        self.setGeometry(100, 100, 1000, 800)  # Увеличил размер окна
        
        self.recommender = AdvancedBookRecommender()
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Установка глобальных стилей для приложения
        self.setStyleSheet("""
            /* Основные стили для всего приложения */
            QMainWindow {
                background-color: #121212;
            }
            
            /* Стили для всех QWidget */
            QWidget {
                background-color: #121212;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            /* Стили для QLabel */
            QLabel {
                color: #e0e0e0;
                font-size: 14px;
            }
            
            /* Стили для QComboBox */
            QComboBox {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
                min-width: 200px;
            }
            
            QComboBox:hover {
                border: 1px solid #555;
            }
            
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #444;
                border-left-style: solid;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #e0e0e0;
                selection-background-color: #3a3a3a;
                outline: none;
            }
            
            /* Стили для QPushButton */
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 120px;
            }
            
            QPushButton:hover {
                background-color: #1565c0;
            }
            
            QPushButton:pressed {
                background-color: #0a3570;
            }
            
            /* Стили для QListWidget */
            QListWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 14px;
            }
            
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #333;
                height: 160px;
            }
            
            QListWidget::item:hover {
                background-color: #2a2a2a;
            }
            
            QListWidget::item:selected {
                background-color: #0d47a1;
            }
            
            /* Стили для QSlider */
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            
            QSlider::sub-page:horizontal {
                background: #0d47a1;
                height: 6px;
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            
            QSlider::handle:horizontal:hover {
                background: #e0e0e0;
            }
            
            /* Стили для заголовка */
            QLabel#title_label {
                font-size: 24px;
                font-weight: bold;
                color: #bb86fc;
                padding: 10px;
            }
            
            /* Стили для сообщений QMessageBox */
            QMessageBox {
                background-color: #1e1e1e;
            }
            
            QMessageBox QLabel {
                color: #e0e0e0;
            }
            
            QMessageBox QPushButton {
                min-width: 80px;
            }
            
            /* Стили для изображений книг */
            .book-cover {
                border-radius: 4px;
                border: 2px solid #555;
                margin-right: 15px;
            }
            
            .book-title {
                font-size: 18px;
                font-weight: bold;
                color: #bb86fc;
                margin-bottom: 5px;
            }
            
            .book-author {
                font-size: 14px;
                color: #a0a0a0;
                margin-bottom: 5px;
            }
            
            .book-genre {
                font-size: 14px;
                color: #7fbcff;
                margin-bottom: 8px;
            }
            
            .book-description {
                font-size: 13px;
                color: #d0d0d0;
                line-height: 1.4;
            }
        """)
        
        main_layout = QVBoxLayout()
        
        # Заголовок
        title_label = QLabel("📖 Лютый книжный рекомендатель")
        title_label.setObjectName("title_label")
        main_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        
        # Выбор книги
        book_layout = QHBoxLayout()
        book_label = QLabel("Выберите книгу:")
        book_layout.addWidget(book_label)
        
        self.book_combo = QComboBox()
        for _, row in self.recommender.books.iterrows():
            self.book_combo.addItem(f"{row['title']} - {row['author']}", row['book_id'])
        book_layout.addWidget(self.book_combo)
        
        # Выбор метода рекомендаций
        method_layout = QHBoxLayout()
        method_label = QLabel("Метод рекомендаций:")
        method_layout.addWidget(method_label)
        
        self.method_combo = QComboBox()
        self.method_combo.addItem("Гибридный (лучший)", "hybrid")
        self.method_combo.addItem("По содержанию", "content")
        self.method_combo.addItem("По оценкам пользователей", "collab")
        self.method_combo.addItem("KNN-рекомендации", "knn")
        method_layout.addWidget(self.method_combo)
        
        # Количество рекомендаций
        count_layout = QHBoxLayout()
        count_label = QLabel("Количество рекомендаций:")
        count_layout.addWidget(count_label)
        
        self.count_slider = QSlider(Qt.Horizontal)
        self.count_slider.setMinimum(3)
        self.count_slider.setMaximum(10)
        self.count_slider.setValue(5)
        self.count_slider.setTickInterval(1)
        self.count_slider.setTickPosition(QSlider.TicksBelow)
        self.count_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #444;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #0d47a1;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
        """)
        count_layout.addWidget(self.count_slider)
        
        self.count_label = QLabel("5")
        count_layout.addWidget(self.count_label)
        
        # Кнопка рекомендаций
        recommend_btn = QPushButton("Получить рекомендации")
        recommend_btn.setStyleSheet("""
            QPushButton {
                background-color: #bb86fc;
                color: #000000;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #9a67ea;
            }
            QPushButton:pressed {
                background-color: #7e57c2;
            }
        """)
        recommend_btn.clicked.connect(self.show_recommendations)
        
        # Список рекомендаций
        self.recommendations_list = QListWidget()
        self.recommendations_list.setIconSize(QSize(100, 150))
        self.recommendations_list.setSpacing(8)
        
        # Сборка интерфейса
        main_layout.addLayout(book_layout)
        main_layout.addLayout(method_layout)
        main_layout.addLayout(count_layout)
        main_layout.addWidget(recommend_btn)
        
        recommendations_label = QLabel("Рекомендуемые книги:")
        recommendations_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(recommendations_label)
        
        main_layout.addWidget(self.recommendations_list)
        
        # Обновление слайдера
        self.count_slider.valueChanged.connect(
            lambda: self.count_label.setText(str(self.count_slider.value())))
        
        central_widget.setLayout(main_layout)
    
    def show_recommendations(self):
        book_id = self.book_combo.currentData()
        method = self.method_combo.currentData()
        top_n = self.count_slider.value()
        
        self.recommendations_list.clear()
        
        try:
            if method == 'knn':
                recommendations = self.recommender.get_knn_recommendations(book_id, top_n)
            else:
                recommendations = self.recommender.get_recommendations(book_id, method, top_n)
            
            if recommendations.empty:
                QMessageBox.information(self, "Информация", "Не удалось получить рекомендации.")
                return
            
            for _, row in recommendations.iterrows():
                # Создаем виджет для отображения информации о книге
                item_widget = QWidget()
                item_layout = QHBoxLayout()
                item_layout.setContentsMargins(10, 10, 10, 10)
                item_layout.setSpacing(15)
                
                # Добавляем изображение обложки
                cover_label = QLabel()
                pixmap = QPixmap()
                pixmap.loadFromData(self.load_image_from_url(self.recommender.book_images[row['book_id']]))
                cover_label.setPixmap(pixmap.scaled(100, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                cover_label.setObjectName("book-cover")
                item_layout.addWidget(cover_label)
                
                # Добавляем информацию о книге
                info_widget = QWidget()
                info_layout = QVBoxLayout()
                info_layout.setContentsMargins(0, 0, 0, 0)
                info_layout.setSpacing(5)
                
                title_label = QLabel(row['title'])
                title_label.setObjectName("book-title")
                
                author_label = QLabel(f"{row['author']}")
                author_label.setObjectName("book-author")
                
                genre_label = QLabel(f"{row['genre']}")
                genre_label.setObjectName("book-genre")
                
                desc_label = QLabel(row['description'])
                desc_label.setObjectName("book-description")
                desc_label.setWordWrap(True)
                
                info_layout.addWidget(title_label)
                info_layout.addWidget(author_label)
                info_layout.addWidget(genre_label)
                info_layout.addWidget(desc_label)
                info_layout.addStretch()
                
                info_widget.setLayout(info_layout)
                item_layout.addWidget(info_widget, stretch=1)
                
                item_widget.setLayout(item_layout)
                
                # Создаем QListWidgetItem и устанавливаем его размер
                list_item = QListWidgetItem()
                list_item.setSizeHint(QSize(900, 160))
                
                # Добавляем виджет в список
                self.recommendations_list.addItem(list_item)
                self.recommendations_list.setItemWidget(list_item, item_widget)
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")
    
    def load_image_from_url(self, url):
        import urllib.request
        with urllib.request.urlopen(url) as response:
            data = response.read()
        return data

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = RecommenderApp()
    window.show()
    sys.exit(app.exec_())
