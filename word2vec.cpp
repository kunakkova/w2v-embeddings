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

clock_t training_start_time = 0;

// Вычисление сигмоиды
double compute_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Генерация случайного веса в заданном диапазоне
double generate_random_weight(double weight_range) {
    static random_device entropy_source;
    static mt19937 generator(entropy_source());
    uniform_real_distribution<double> distribution(-weight_range, weight_range);
    return distribution(generator);
}

// Функция для загрузки текстовых данных, предобработки и построения словаря
vector<vector<int>> load_and_preprocess_corpus(const string &corpus_file, unordered_map<string, int> &vocabulary_map, vector<string> &reverse_vocabulary, int max_vocab_size
) {
    ifstream input_file(corpus_file);
    if (!input_file) {
        cerr << "Ошибка открытия файла с корпусом!" << '\n';
        exit(1);
    }

    unordered_map<string, int> word_frequency; // Частота встречаемости слов
    vector<vector<string>> tokenized_sentences; // Токенизированные предложения
    string text_line;

    // Чтение файла построчно
    while (getline(input_file, text_line)) {
        replace(text_line.begin(), text_line.end(), '\n', ' ');
        replace(text_line.begin(), text_line.end(), '\r', ' ');
        
        // Разбиение строки на отдельные слова
        istringstream line_stream(text_line);
        vector<string> sentence_tokens(
            (istream_iterator<string>(line_stream)), 
            istream_iterator<string>()
        );
        
        // Очистка и нормализация каждого токена в предложении
        for (string &token : sentence_tokens) {
            // Удаление пунктуации
            token.erase(remove_if(token.begin(), token.end(), 
                        [](unsigned char c) { return ispunct(c); }), 
                        token.end());
            
            // Приведение к нижнему регистру
            transform(token.begin(), token.end(), token.begin(),
                     [](unsigned char c) { return tolower(c); });
            
            if (token.empty()) continue;
        }
        
        // Удаление пустых токенов из всего предложения
        sentence_tokens.erase(
            remove_if(sentence_tokens.begin(), sentence_tokens.end(), 
                   [](const string &w) { return w.empty(); }), 
            sentence_tokens.end()
        );
        
        // Обновление частот слов
        if (!sentence_tokens.empty()) {
            tokenized_sentences.push_back(sentence_tokens);
            for (const string &word : sentence_tokens) {
                word_frequency[word]++;
            }
        }
    }
    input_file.close();

    vector<pair<int, string>> sorted_vocabulary;
    for (auto &entry : word_frequency) {
        // Сохраняем пары (частота, слово) для сортировки
        sorted_vocabulary.push_back({entry.second, entry.first});
    }

    cout << "Обнаружено уникальных слов: " << sorted_vocabulary.size() << '\n';
    // Сортировка от наибольшей к наименьшей частоте
    sort(sorted_vocabulary.rbegin(), sorted_vocabulary.rend());

    // Ограничение словаря до заданного размера
    if (sorted_vocabulary.size() > (size_t)max_vocab_size) {
        // Обрезаем вектор до max_vocab_size элементов
        sorted_vocabulary.resize(max_vocab_size);
    }

    // Построение взаимных отображений между словами и индексами
    for (size_t i = 0; i < sorted_vocabulary.size(); i++) {
        const string &word = sorted_vocabulary[i].second;
        const int index = i;
        vocabulary_map[word] = index;          // Слово -> индекс
        reverse_vocabulary.push_back(word);    // Индекс -> слово
    }

    cout << "Фактический размер словаря: " << vocabulary_map.size() << '\n';

    // Преобразование текста в последовательности числовых индексов
    vector<vector<int>> encoded_sentences;
    for (const auto &sentence : tokenized_sentences) {
        vector<int> encoded_sentence;
        for (const string &word : sentence) {
            if (vocabulary_map.count(word)) {
                // Заменяем слово его индексом из словаря
                encoded_sentence.push_back(vocabulary_map[word]);
            }
        }
        if (!encoded_sentence.empty()) {
            encoded_sentences.push_back(encoded_sentence);
        }
    }

    return encoded_sentences;
}
// Word2Vec
void train_word_embeddings(const vector<vector<int>> &encoded_sentences, int vocabulary_size, int embedding_dim, int context_window, int negative_samples, int training_epochs, float learning_rate, vector<vector<double>> &embedding_matrix
) {
    // Инициализация матрицы эмбеддингов случайными значениями
    embedding_matrix.assign(vocabulary_size, vector<double>(embedding_dim));
    const double init_range = 0.5 / embedding_dim;
    
    for (int word_idx = 0; word_idx < vocabulary_size; word_idx++) {
        for (int dim = 0; dim < embedding_dim; dim++) {
            embedding_matrix[word_idx][dim] = generate_random_weight(init_range);
        }
    }

    // Настройка генератора случайных чисел для отрицательного сэмплирования
    random_device entropy_source;
    mt19937 generator(entropy_source());
    uniform_int_distribution<int> negative_sampler(0, vocabulary_size - 1);

    // Основной цикл обучения
    for (int epoch = 0; epoch < training_epochs; epoch++) {
        auto epoch_start = clock();
        double epoch_loss = 0.0;
        int processed_pairs = 0;

        // Итерация по всем предложениям корпуса
        for (const auto &sentence : encoded_sentences) {
            int sentence_length = sentence.size();
            
            // Обработка каждого слова в предложении как целевого
            for (int target_pos = 0; target_pos < sentence_length; target_pos++) {
                int target_word_idx = sentence[target_pos]; // Индекс текущего целевого слова

                // Определение границ контекстного окна
                // [target_pos - window, target_pos + window] 
                int window_start = max(0, target_pos - context_window);
                int window_end = min(sentence_length, target_pos + context_window + 1);
                
                // Итерация по словам в контекстном окне
                for (int context_pos = window_start; context_pos < window_end; context_pos++) {
                    if (context_pos == target_pos) continue; // Пропуск  целевого слова
                    int context_word_idx = sentence[context_pos]; // Индекс слова контекста

                    /*** Обучение на положительном примере (target + context) ***/
                    
                    // Вычисление скалярного произведения векторов (v_target * v_context)
                    double dot_product = 0.0;
                    for (int dim = 0; dim < embedding_dim; dim++) {
                        dot_product += embedding_matrix[target_word_idx][dim] * 
                                      embedding_matrix[context_word_idx][dim];
                    }
                    
                    // Преобразование в вероятность с помощью сигмоиды
                    double positive_prob = compute_sigmoid(dot_product);
                    
                    double prediction_error = positive_prob - 1.0;
                    
                    // Накопление значения функции потерь: -log(sig(v*u))
                    epoch_loss += -log(positive_prob + 1e-8); // +1e-8 для численной стабильности

                    // Обновление векторов через градиентный спуск
                    // v_target = v_target - lr * (sig(v*u) - 1) * v_context
                    // v_context = v_context - lr * (sig(v*u) - 1) * v_target
                    for (int dim = 0; dim < embedding_dim; dim++) {
                        double target_cache = embedding_matrix[target_word_idx][dim];
                        embedding_matrix[target_word_idx][dim] -= learning_rate * prediction_error * 
                                                                embedding_matrix[context_word_idx][dim];
                        embedding_matrix[context_word_idx][dim] -= learning_rate * prediction_error * 
                                                                 target_cache;
                    }

                    /*** Обучение на негативных примерах (target + random words) ***/
                    for (int sample = 0; sample < negative_samples; sample++) {
                        int negative_word_idx = negative_sampler(generator); // Случайное слово из словаря
                        
                        // Вычисление скалярного произведения с негативным примером
                        double negative_dot = 0.0;
                        for (int dim = 0; dim < embedding_dim; dim++) {
                            negative_dot += embedding_matrix[target_word_idx][dim] * 
                                         embedding_matrix[negative_word_idx][dim];
                        }
                        
                        // Преобразование в вероятность: sig(v*u_neg)
                        double negative_prob = compute_sigmoid(negative_dot);
                        
                        // Ошибка для негативного примера: sig(v*u_neg) - 0 (целевое значение = 0)
                        double negative_error = negative_prob;
                        
                        // Накопление значения функции потерь: -log(1 - sig(v*u_neg))
                        epoch_loss += -log(1.0 - negative_prob + 1e-8);

                        // Обновление векторов через градиентный спуск
                        // v_target = v_target - lr * sig(v*u_neg) * v_neg
                        // v_neg = v_neg - lr * sig(v*u_neg) * v_target
                        for (int dim = 0; dim < embedding_dim; dim++) {
                            double target_cache = embedding_matrix[target_word_idx][dim];
                            embedding_matrix[target_word_idx][dim] -= learning_rate * negative_error * 
                                                                    embedding_matrix[negative_word_idx][dim];
                            embedding_matrix[negative_word_idx][dim] -= learning_rate * negative_error * 
                                                                      target_cache;
                        }
                    }
                    processed_pairs++; // Учет обработанной пары (1 позитивная + k негативных)
                }
            }
        }
        
        cout << "Эпоха " << epoch + 1 
             << " | Средняя ошибка: " << epoch_loss / processed_pairs
             << " | Длительность: " << double(clock() - epoch_start) / CLOCKS_PER_SEC 
             << " сек" << endl;
    }
}

// Сохранение обученных векторных представлений
void save_word_vectors(
    const string &output_file,
    const vector<vector<double>> &embedding_matrix,
    const vector<string> &reverse_vocabulary
) {
    ofstream output_stream(output_file);
    output_stream << reverse_vocabulary.size() << " " << embedding_matrix[0].size() << "\n";
    for (size_t word_idx = 0; word_idx < reverse_vocabulary.size(); word_idx++) {
        output_stream << reverse_vocabulary[word_idx];
        for (double value : embedding_matrix[word_idx]) {
            output_stream << " " << value;
        }
        output_stream << "\n";
    }
}

int main() {
    const string corpus_path = "books.txt";
    const string embeddings_output = "embeddings_1000.txt";
    const int embedding_dimension = 1000;
    const int vocabulary_limit = 50000;
    const int context_window_size = 4;
    const int negative_samples_count = 5;
    const int total_epochs = 5;
    const float initial_learning_rate = 0.025;

    training_start_time = clock();

    // Инициализация структур данных
    unordered_map<string, int> word_to_index;
    vector<string> index_to_word;
    vector<vector<int>> processed_sentences = load_and_preprocess_corpus(
        corpus_path, 
        word_to_index, 
        index_to_word, 
        vocabulary_limit
    );

    // Обучение модели
    vector<vector<double>> embeddings;
    train_word_embeddings(
        processed_sentences,
        vocabulary_limit,
        embedding_dimension,
        context_window_size,
        negative_samples_count,
        total_epochs,
        initial_learning_rate,
        embeddings
    );

    // Сохранение результатов
    save_word_vectors(embeddings_output, embeddings, index_to_word);

    auto training_end = clock();
    double total_time = double(training_end - training_start_time) / CLOCKS_PER_SEC;
    cout << "Обучение завершено за " << total_time << " секунд. Результаты сохранены в " 
         << embeddings_output << endl;
    return 0;
}
