#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;

clock_t start_time = 0;

// Функция сигмоиды
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Функция для генерации случайного числа в диапазоне [-range, range).
double random_double(double range) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<double> dist(-range, range);
    return dist(gen);
}

// Загружает текст из файла, создаёт словарь и кодирует предложения в последовательность индексов.
// Индексы присваиваются по частоте слова в тексте с ограничением на размер словаря
vector<vector<int>> load_text(const string &filename, unordered_map<string, int> &word2index, vector<string> &index2word, int vocab_size) {
    ifstream file(filename);
    if (!file) {
        cerr << "Ошибка при открытии файла!" << '\n';
        exit(1);
    }

    // word_freq будет хранить сколько раз встречается каждое слово
    unordered_map<string, int> word_freq;
    // raw_sentences хранит предложения в виде векторов строк
    vector<vector<string>> raw_sentences;
    string line;

    // Читаем файл построчно
    while (getline(file, line)) {
        // Заменяем символы новой строки на пробелы
        replace(line.begin(), line.end(), '\n', ' ');
        replace(line.begin(), line.end(), '\r', ' ');
        
        istringstream iss(line);
        vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>());
        
        for (string &word : words) {
            // Удаляем пунктуацию
            word.erase(remove_if(word.begin(), word.end(), 
                      [](unsigned char c) { return ispunct(c); }), 
                      word.end());
            
            // Приводим к нижнему регистру
            transform(word.begin(), word.end(), word.begin(),
                     [](unsigned char c) { return tolower(c); });
            
            // Удаляем пустые строки после обработки
            if (word.empty()) {
                continue;
            }
        }
        
        // Удаляем пустые слова из предложения
        words.erase(remove_if(words.begin(), words.end(), 
                   [](const string &w) { return w.empty(); }), 
                   words.end());
        
        if (!words.empty()) {
            raw_sentences.push_back(words);
            for (const string &word : words) {
                word_freq[word]++;
            }
        }
    }
    file.close();

    // Создаем вектор пар (частота, слово) для сортировки по частоте.
    vector<pair<int, string>> sorted_words;
    for (auto &p : word_freq) {
        // if (p.second > 2)  // Отфильтровываем редкие слова (частота > 2)
        sorted_words.push_back({p.second, p.first});
    }

    cout << "Количество уникальных слов: " << sorted_words.size() << '\n';
    cout << endl;
    // Сортировка по убыванию частоты
    sort(sorted_words.rbegin(), sorted_words.rend());

    // Ограничиваем словарь до указанного размера
    if (sorted_words.size() > (size_t)vocab_size) sorted_words.resize(vocab_size);

    // Заполняем отображения: word2index и index2word
    for (size_t i = 0; i < sorted_words.size(); i++) {
        word2index[sorted_words[i].second] = i;
        index2word.push_back(sorted_words[i].second);
    }

    cout << "Размер словаря: " << word2index.size() << '\n';
    if (word2index.size() == vocab_size) cout << "[WARNING] Словарь заполнен до максимального размера." << '\n';

    // Кодируем предложения: каждое слово заменяется на свой индекс, если оно есть в словаре
    vector<vector<int>> sentences;
    for (const auto &sentence : raw_sentences) {
        vector<int> encoded;
        for (const string &word : sentence)
            if (word2index.count(word)) encoded.push_back(word2index[word]);
        if (!encoded.empty()) sentences.push_back(encoded);
    }

    return sentences;
}

// Обучение эмбеддингов
void train_word2vec(const vector<vector<int>> &sentences, int vocab_size, int d, int window_size, int k, int epochs, double lr,
                    vector<vector<double>> &word_vectors) {
    // Инициализируем матрицу эмбеддингов: для каждого слова в словаре выделяем вектор размерности d
    word_vectors.assign(vocab_size, vector<double>(d));

    // Инициализация эмбеддингов случайными значениями в диапазоне [-0.5/d, 0.5/d]
    for (int i = 0; i < vocab_size; i++)
        for (int j = 0; j < d; j++) word_vectors[i][j] = random_double(0.5 / d);

    // Настраиваем генератор случайных чисел для отрицательного сэмплинга
    random_device rd;
    mt19937 gen(rd());
    // Используем равномерное распределение по индексам от 0 до vocab_size-1 для выбора отрицательных примеров
    uniform_int_distribution<int> neg_sample(0, vocab_size - 1);

    // Основной цикл по эпохам обучения
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto start_epoch = clock();
        double total_loss = 0.0;
        int count = 0;  // Счётчик обработанных примеров

        // Проходим по каждому предложению
        for (const auto &sentence : sentences) {
            int n = sentence.size();
            // Проходим по каждому слову в предложении
            for (int i = 0; i < n; i++) {
                int target = sentence[i];  // Текущее целевое слово

                // Определяем контекстное окно для текущего слова:
                // слова от max(0, i - window_size) до min(n, i + window_size + 1) / влево и вправо на window_size
                for (int j = max(0, i - window_size); j < min(n, i + window_size + 1); j++) {
                    if (j == i) continue;  // Пропускаем само целевое слово
                    int context = sentence[j];

                    // Обработка положительного примера

                    // Вычисляем скалярное произведение целевого и контекстного векторов:
                    // dot = sum(v_target[z] * v_context[z]) по z от 0 до d-1.
                    double dot = 0.0;
                    for (int z = 0; z < d; z++) dot += word_vectors[target][z] * word_vectors[context][z];

                    // Применяем сигмоиду для получения вероятности:
                    // pos_pred = sig(dot)
                    double pos_pred = sigmoid(dot);
                    // Градиент для положительного примера grad = (sig(dot) - 1)  (истинное значение = 1)
                    double grad = pos_pred - 1.0;
                    // Лосс для положительного примера: -log(sig(dot))
                    total_loss += -log(pos_pred + 1e-8);

                    // Обновляем векторы для положительного примера:
                    // v_target = v_target - lr * grad * v_context
                    // v_context = v_context - lr * grad * v_target (старое значение v_target)
                    for (int z = 0; z < d; z++) {
                        double temp = word_vectors[target][z];
                        word_vectors[target][z] -= lr * grad * word_vectors[context][z];
                        word_vectors[context][z] -= lr * grad * temp;
                    }

                    // Обработка отрицательных примеров

                    // Для одного положительного примера выбираем k отрицательных примеров
                    for (int neg = 0; neg < k; neg++) {
                        // Выбираем случайное слово из словаря как отрицательный пример
                        int neg_word = neg_sample(gen);
                        // Вычисляем dot для отрицательной пары (target, neg_word)
                        double dot_neg = 0.0;
                        for (int z = 0; z < d; z++) dot_neg += word_vectors[target][z] * word_vectors[neg_word][z];

                        // Применяем сигмоиду: получаем вероятность того, что пара встречается
                        double neg_pred = sigmoid(dot_neg);
                        // Лосс для отрицательного примера: -log(1 - sig(dot_neg))
                        total_loss += -log(1.0 - neg_pred + 1e-8);
                        // Градиент для отрицательного примера sig(dot_neg), истинное значение = 0
                        double neg_grad = neg_pred;

                        // Обновляем векторы для отрицательного примера
                        for (int z = 0; z < d; z++) {
                            double temp = word_vectors[target][z];
                            word_vectors[target][z] -= lr * neg_grad * word_vectors[neg_word][z];
                            word_vectors[neg_word][z] -= lr * neg_grad * temp;
                        }
                    }
                    // Увеличиваем счётчик обработанных примеров (каждая пара считается одним примером)
                    count++;
                }
            }
        }
        // Вывод средней ошибки за эпоху, время выполнения эпохи и общее время с начала обучения
        cout << "Эпоха " << epoch + 1 << " | Средняя ошибка: " << total_loss / count << " | Время: " << double(clock() - start_epoch) / CLOCKS_PER_SEC << " сек"
             << " сек" << endl;
    }
}

// Сохраняет обученные эмбеддинги в файл.
//   vocab_size d – количество слов и размерность векторов.
//   vocab_size слов и соответсвующих векторов
void save_embeddings(const string &filename, const vector<vector<double>> &embeddings, const vector<string> &index2word) {
    ofstream ofs(filename);
    ofs << index2word.size() << " " << embeddings[0].size() << "\n";
    for (size_t i = 0; i < index2word.size(); i++) {
        ofs << index2word[i];
        for (double val : embeddings[i]) ofs << " " << val;
        ofs << "\n";
    }
}

int main() {
    string sentences_path = "books.txt";
    string embeddings_path = "embeddings_500.txt";
    int d = 500;

    int vocab_size = 50000;
    int window_size = 4;
    int k = 5;
    int epochs = 5;
    double lr = 0.025;

    start_time = clock();

    // Создаем word2index и index2word, и кодируем предложения
    unordered_map<string, int> word2index;
    vector<string> index2word;
    vector<vector<int>> sentences = load_text(sentences_path, word2index, index2word, vocab_size);

    // Обучаем эмбеддинги
    vector<vector<double>> word_vectors;
    train_word2vec(sentences, vocab_size, d, window_size, k, epochs, lr, word_vectors);

    // Сохраняем обученные эмбеддинги в файл
    save_embeddings(embeddings_path, word_vectors, index2word);

    auto end_time = clock();
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Обучение завершено. Эмбеддинги сохранены в " << embeddings_path << endl;
    return 0;
}
