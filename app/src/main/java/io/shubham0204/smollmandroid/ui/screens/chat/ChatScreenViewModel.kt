/*
 * Copyright (C) 2024 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.shubham0204.smollmandroid.ui.screens.chat

import android.app.Application
import android.content.Context
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import android.util.TypedValue
import androidx.compose.runtime.mutableStateOf
import androidx.core.content.res.ResourcesCompat
import androidx.lifecycle.AndroidViewModel
// import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.github.jelmerk.knn.DistanceFunctions
import com.github.jelmerk.knn.hnsw.HnswIndex
import com.ml.shubham0204.sentence_embeddings.SentenceEmbedding
import io.noties.markwon.AbstractMarkwonPlugin
import io.noties.markwon.Markwon
import io.noties.markwon.core.CorePlugin
import io.noties.markwon.core.MarkwonTheme
import io.noties.markwon.syntax.Prism4jThemeDarkula
import io.noties.markwon.syntax.SyntaxHighlightPlugin
import io.noties.prism4j.Prism4j
import io.shubham0204.smollm.SmolLM
import io.shubham0204.smollmandroid.R
import io.shubham0204.smollmandroid.data.Chat
import io.shubham0204.smollmandroid.data.ChatMessage
import io.shubham0204.smollmandroid.data.ChatsDB
import io.shubham0204.smollmandroid.data.MessagesDB
import io.shubham0204.smollmandroid.data.TasksDB
import io.shubham0204.smollmandroid.llm.ModelsRepository
import io.shubham0204.smollmandroid.prism4j.PrismGrammarLocator
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.apache.commons.csv.CSVFormat
import org.koin.android.annotation.KoinViewModel
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Date
import kotlin.concurrent.thread

const val LOGTAG = "[ReggaeLLM]"
val LOGD: (String) -> Unit = { Log.d(LOGTAG, it) }

@KoinViewModel
class ChatScreenViewModel(
    private val application: Application,
    val context: Context,
    val messagesDB: MessagesDB,
    val chatsDB: ChatsDB,
    val modelsRepository: ModelsRepository,
    val tasksDB: TasksDB,
) : AndroidViewModel(application) {
    val smolLM = SmolLM()

    val currChatState = mutableStateOf<Chat?>(null)

    val isGeneratingResponse = mutableStateOf(false)
    val partialResponse = mutableStateOf("")

    val showSelectModelListDialogState = mutableStateOf(false)
    val showMoreOptionsPopupState = mutableStateOf(false)
    val showTaskListBottomListState = mutableStateOf(false)

    val isInitializingModel = mutableStateOf(false)
    var responseGenerationJob: Job? = null

    // private val application = getApplication<Application>()
    private val appDirFile = application.getExternalFilesDir("")
    private var showAlert = mutableStateOf(false)
    private var alertMessage = mutableStateOf("")
    private var hnswIndex = HnswIndex
        .newBuilder<FloatArray, Float>(1, DistanceFunctions.FLOAT_INNER_PRODUCT, 1)
        .build<String, Chunk>()

    companion object {
        // const val AppConfigFilename = "mlc-app-config.json"
        // const val ModelConfigFilename = "mlc-chat-config.json"
        // const val ParamsConfigFilename = "ndarray-cache.json"
        const val VectorsFilename = "embeddings.csv"
        const val ChunksFilename = "chunks.csv"
        const val IndexFilename = "doc_index.bin"
        // const val ModelUrlSuffix = "resolve/main/"
        const val EmbedFilename = "bge-small-en-v1.5.onnx"
        const val TokenizerFilename = "tokenizer.json"
    }

    val embedder = SentenceEmbedding()
    val embedModel = application.assets.open(EmbedFilename)
    val tokenFile = application.assets.open(TokenizerFilename)
    val tokenBytes = tokenFile.readBytes()

    val markwon: Markwon

    init {
        currChatState.value = chatsDB.loadDefaultChat()
        val prism4j = Prism4j(PrismGrammarLocator())
        markwon =
            Markwon
                .builder(context)
                .usePlugin(CorePlugin.create())
                .usePlugin(SyntaxHighlightPlugin.create(prism4j, Prism4jThemeDarkula.create()))
                .usePlugin(
                    object : AbstractMarkwonPlugin() {
                        override fun configureTheme(builder: MarkwonTheme.Builder) {
                            val jetbrainsMonoFont =
                                ResourcesCompat.getFont(context, R.font.jetbrains_mono)!!
                            builder
                                .codeBlockTypeface(
                                    ResourcesCompat.getFont(context, R.font.jetbrains_mono)!!,
                                ).codeBlockTextColor(Color.WHITE)
                                .codeBlockTextSize(spToPx(10f))
                                .codeBlockBackgroundColor(Color.BLACK)
                                .codeTypeface(jetbrainsMonoFont)
                                .codeTextSize(spToPx(10f))
                                .codeTextColor(Color.WHITE)
                                .codeBackgroundColor(Color.BLACK)
                        }
                    },
                ).build()
        var embedFile : File? = null
        var f : FileOutputStream? = null
        try {
            embedFile = File(context.filesDir, EmbedFilename)
            f = FileOutputStream(embedFile)
            val buffer = ByteArray(1024)
             var read : Int? = null
            while (embedModel.read(buffer).also({ read = it }) != -1) {
                read?.let { f.write(buffer, 0, it) }
            }
        } catch (e: IOException) {
            Log.e("[Embedder]", "Failed to store model file")
        } finally {
            embedModel.close()
            if (f != null) {
                try {
                    f.close()
                } catch (e: IOException) {
                    e.printStackTrace()
                }
            }
        }
        CoroutineScope(Dispatchers.Default).launch {
            embedder.init(
                modelFilepath = embedFile!!.absolutePath,
                tokenizerBytes = tokenBytes,
                useTokenTypeIds = true,
                outputTensorName = "sentence_embedding",
                useFP16 = false,
                useXNNPack = false,
                normalizeEmbeddings = false
            )
        }
        if (!isIndexReady()) {
            CoroutineScope(Dispatchers.Default).launch {
                prepareRetrievalData()
            }
        } else {
            loadRetrievalData()
        }
    }

    private fun spToPx(sp: Float): Int =
        TypedValue
            .applyDimension(TypedValue.COMPLEX_UNIT_SP, sp, context.resources.displayMetrics)
            .toInt()

    private fun issueAlert(error: String) {
        showAlert.value = true
        alertMessage.value = error
    }

    fun isIndexReady(): Boolean {
        val indexFile = File(appDirFile, IndexFilename)
        return indexFile.exists()
    }

    private fun prepareRetrievalData() {
        try {
            val reader1 = application.assets.open(VectorsFilename).bufferedReader()
            val csvFormat1 = CSVFormat.DEFAULT
                .builder()
                .setIgnoreSurroundingSpaces(true)
                .setIgnoreEmptyLines(true)
                .build()
            val records1 = csvFormat1.parse(reader1).drop(1)
            val reader2 = application.assets.open(ChunksFilename).bufferedReader()
            val csvFormat2 = CSVFormat.DEFAULT
                .builder()
                .setIgnoreSurroundingSpaces(true)
                .setIgnoreEmptyLines(true)
                .setDelimiter('|')
                .build()
            val records2 = csvFormat2.parse(reader2).drop(1)

            var chunks = emptyList<Chunk>()
            for (i in records2.indices) {
                val chunk =
                    Chunk(records2[i][0], records1[i][0].split("\t").map { it.toFloat() }.toFloatArray())
                chunks = chunks.plusElement(chunk)
            }

            CoroutineScope(Dispatchers.Default).launch {
                val vector1 = records1[0][0].split("\t").map { it.toFloat() }.toFloatArray()
                val vector2 = embedder.encode(records2[0][0])

                Log.i("STUFF", "vector 1: " + records1[0][0])
                Log.i("STUFF", "chunk 1: " + records2[0][0])
                Log.i("STUFF", "vector from database: " + vector1.contentToString())
                Log.i("STUFF", "generated vector: " + vector2.contentToString())
                Log.i("STUFF", "Are embeddings the same? " + (vector1.contentEquals(vector2)).toString())
            }

            hnswIndex = HnswIndex
                .newBuilder<FloatArray, Float>(
                    chunks[0].vector().size,
                    DistanceFunctions.FLOAT_INNER_PRODUCT,
                    chunks.size
                )
                .withM(4)
                .withEf(16)
                .withEfConstruction(16)
                .build<String, Chunk>()

            for (i in chunks.indices) {
                hnswIndex.add(chunks[i])
            }

            hnswIndex.save(File(appDirFile, IndexFilename))

            if (!File(appDirFile, IndexFilename).exists()) {
                throw Exception("Saving index failed")
            }

        } catch (e: Exception) {
            viewModelScope.launch {
                issueAlert("Index creation failed: ${e.message}")
            }
        }
    }

    private fun loadRetrievalData() {
        if (!File(appDirFile, IndexFilename).exists()) {
            throw Exception("Index file not found")
        }
        hnswIndex = HnswIndex.load(File(appDirFile, IndexFilename))
    }

    fun getChats(): Flow<List<Chat>> = chatsDB.getChats()

    fun getChatMessages(): Flow<List<ChatMessage>>? {
        return currChatState.value?.let { chat ->
            return messagesDB.getMessages(chat.id)
        }
    }

    fun updateChatLLM(modelId: Long) {
        currChatState.value = currChatState.value?.copy(llmModelId = modelId)
        chatsDB.updateChat(currChatState.value!!)
    }

    fun updateChat(chat: Chat) {
        currChatState.value = chat
        chatsDB.updateChat(chat)
        loadModel()
    }

    fun sendUserQuery(query: String) {
        currChatState.value?.let { chat ->
            chat.dateUsed = Date()
            chatsDB.updateChat(chat)
            if (chat.isTask) {
                messagesDB.deleteMessages(chat.id)
            }

            responseGenerationJob =
                CoroutineScope(Dispatchers.Default).launch {
                    var startTime = SystemClock.uptimeMillis()
                    val embedding: FloatArray = embedder.encode(query)
                    val embedTime = SystemClock.uptimeMillis() - startTime

                    startTime = SystemClock.uptimeMillis()
                    val retrievedInfo = hnswIndex.findNearest(embedding, 2)
                    val retrievalTime = SystemClock.uptimeMillis() - startTime

                    Log.i("STUFF", "database size: " + hnswIndex.size())
                    for (elem in retrievedInfo) {
                        val str = elem.item().id()
                        Log.i("STUFF", "retrievedInfo: " + str)
                    }

                    Log.i("[TextEmbedder]", "Query: " + query)
                    Log.i("[TextEmbedder]", "Embedding: " + embedding.contentToString())
                    Log.i("[EmbedRuntime]", "Runtime: " + embedTime.toString())
                    Log.i("[RetrievalRuntime]", "Runtime: " + retrievalTime.toString())

                    var chatMsg = "Retrieved contexts:"
                    var final_prompt = "You are an agent that guides the user with a step-by-step guide.\n" +
                            "Use the context provided and if you don't have fitting info simply state you can't answer the question.\n" +
                            "Do not use all of the context! Filter out only what you need.\n" +
                            "Absorb the context as if it were your own knowledge.\n" +
                            "Please, use the following context to answer the user query:"

                    for ((idx, result) in retrievedInfo.withIndex()) {
                        final_prompt += ("\n" + idx.toString() + ": " + result.item().id() + ";")
                        chatMsg += ("\n" + idx.toString() + ": " + result.item().id() + ";")
                    }

                    final_prompt += "\nAnd here is the user query:\n$query"
                    chatMsg += "\n" + "Prompt:\n$query"
                    Log.i("[PROMPT]", chatMsg)

                    messagesDB.addUserMessage(chat.id, chatMsg)
                    isGeneratingResponse.value = true

                    partialResponse.value = ""

                    // startTime = SystemClock.uptimeMillis()
                    smolLM.getResponse(final_prompt).collect { partialResponse.value += it }
                    // val genTime = (SystemClock.uptimeMillis() - startTime) / 1000
                    // Log.i("[SmolChat]", "Response generation runtime (s): " + genTime.toString())

                    messagesDB.addAssistantMessage(chat.id, partialResponse.value)
                    withContext(Dispatchers.Main) { isGeneratingResponse.value = false }
                }
        }
    }

    fun stopGeneration() {
        isGeneratingResponse.value = false
        responseGenerationJob?.let { job ->
            if (job.isActive) {
                job.cancel()
            }
        }
    }

    fun switchChat(chat: Chat) {
        stopGeneration()
        currChatState.value = chat
    }

    fun deleteChat(chat: Chat) {
        stopGeneration()
        chatsDB.deleteChat(chat)
        messagesDB.deleteMessages(chat.id)
        currChatState.value = null
    }

    fun deleteModel(modelId: Long) {
        modelsRepository.deleteModel(modelId)
        if (currChatState.value?.llmModelId == modelId) {
            currChatState.value = currChatState.value?.copy(llmModelId = -1)
            smolLM.close()
        }
    }

    /**
     * Load the model for the current chat. If chat is configured with a LLM (i.e. chat.llModelId !=
     * -1), then load the model. If not, show the model list dialog. Once the model is finalized,
     * read the system prompt and user messages from the database and add them to the model.
     */
    fun loadModel() {
        currChatState.value?.let { chat ->
            if (chat.llmModelId == -1L) {
                showSelectModelListDialogState.value = true
            } else {
                val model = modelsRepository.getModelFromId(chat.llmModelId)
                if (model != null) {
                    isInitializingModel.value = true
                    CoroutineScope(Dispatchers.Default).launch {
                        smolLM.create(model.path, chat.minP, chat.temperature, !chat.isTask)
                        LOGD("Model loaded")
                        if (chat.systemPrompt.isNotEmpty()) {
                            smolLM.addSystemPrompt(chat.systemPrompt)
                            LOGD("System prompt added")
                        }
                        if (!chat.isTask) {
                            messagesDB.getMessagesForModel(chat.id).forEach { message ->
                                if (message.isUserMessage) {
                                    smolLM.addUserMessage(message.message)
                                    LOGD("User message added: ${message.message}")
                                } else {
                                    smolLM.addAssistantMessage(message.message)
                                    LOGD("Assistant message added: ${message.message}")
                                }
                            }
                        }
                        withContext(Dispatchers.Main) { isInitializingModel.value = false }
                    }
                } else {
                    showSelectModelListDialogState.value = true
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        smolLM.close()
    }
}
