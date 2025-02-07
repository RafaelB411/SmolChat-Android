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

package io.shubham0204.smollm

import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext

class SmolLM {
    private var nativePtr = 0L

    companion object {
        init {
            System.loadLibrary("smollm")
        }
    }

    suspend fun create(
        modelPath: String,
        minP: Float = 0.05f,
        temperature: Float = 1.0f,
        storeChats: Boolean = false,
    ) = withContext(Dispatchers.IO) {
        nativePtr = loadModel(modelPath, minP, temperature, storeChats)
    }

    fun addUserMessage(message: String) {
        assert(nativePtr != 0L) { "Model is not loaded. Use SmolLM.create to load the model" }
        addChatMessage(nativePtr, message, "user")
    }

    fun addSystemPrompt(prompt: String) {
        assert(nativePtr != 0L) { "Model is not loaded. Use SmolLM.create to load the model" }
        addChatMessage(nativePtr, prompt, "system")
    }

    fun addAssistantMessage(message: String) {
        assert(nativePtr != 0L) { "Model is not loaded. Use SmolLM.create to load the model" }
        addChatMessage(nativePtr, message, "assistant")
    }

    fun getResponse(query: String): Flow<String> =
        flow {
            assert(nativePtr != 0L) { "Model is not loaded. Use SmolLM.create to load the model" }
            var tokens = 0
            startCompletion(nativePtr, query)
            val startTime = SystemClock.uptimeMillis()
            var piece = completionLoop(nativePtr)
            val genTime = SystemClock.uptimeMillis() - startTime
            Log.i("[SmolChat]", "Response generation time (s): " + (genTime/1000).toString())
            while (piece != "[EOG]") {
                tokens += 1
                emit(piece)
                piece = completionLoop(nativePtr)
            }
            val chatTime = SystemClock.uptimeMillis() - startTime
            Log.i("[SmolChat]", "Total chat response time (s): " + (chatTime/1000).toString())
            Log.i("[SmolChat]", "Amount of tokens generated: " +  tokens)
            stopCompletion(nativePtr)
        }

    fun close() {
        close(nativePtr)
    }

    private external fun loadModel(
        modelPath: String,
        minP: Float,
        temperature: Float,
        storeChats: Boolean,
    ): Long

    private external fun addChatMessage(
        modelPtr: Long,
        message: String,
        role: String,
    )

    private external fun close(modelPtr: Long)

    private external fun startCompletion(
        modelPtr: Long,
        prompt: String,
    )

    private external fun completionLoop(modelPtr: Long): String

    private external fun stopCompletion(modelPtr: Long)
}
