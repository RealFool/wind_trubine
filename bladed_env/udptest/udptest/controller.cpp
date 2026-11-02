// Windows 原生 UDP 头文件（必须在 windows.h 之前）
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <tlhelp32.h>
#pragma comment(lib, "ws2_32.lib")

#include <string>
#include <vector>
#include <exception>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cwchar>
#include <fstream>
#include <io.h>
#include <limits>

#include "ExternalControllerApi.h"

using namespace GHTurbineInterface;


namespace
{
    enum UdpFieldIndex : int
    {
        kTime = 0,
        kRotorAzimuth,
        kRotorSpeed,
        kYawError,
        kElectricalTorque,
        kTorqueDemand,
        kGeneratorSpeed,
        kPitchPosition1,
        kPitchDemand1,
        kPitchPosition2,
        kPitchPosition3,
        kNacelleForeAftAcceleration,
        kNacelleSideSideAcceleration,
        kAverageWindSpeed,
        kElectricalPower,
        kRealPower,
        kBladeRoot1My,
        kBladeRoot2My,
        kBladeRoot3My,
        kBladeRoot1Mx,
        kBladeRoot2Mx,
        kBladeRoot3Mx,
    };

    constexpr int LEGACY_UDP_FIELD_COUNT = kBladeRoot3My + 1;
    constexpr int UDP_FIELD_COUNT = kBladeRoot3Mx + 1;
    static_assert(LEGACY_UDP_FIELD_COUNT == 19, "The legacy UDP schema must contain 19 fields.");
    static_assert(UDP_FIELD_COUNT == 22, "The extended UDP schema must contain 22 fields.");

    // Defaults for the offline VirtualBox Host-only network. Environment
    // variables still take precedence when a different endpoint is needed.
    constexpr const char* DEFAULT_UDP_SERVER_IP = "192.168.56.1";
    constexpr int DEFAULT_UDP_SERVER_PORT = 8888;
    constexpr DWORD DEFAULT_UDP_RECEIVE_TIMEOUT_MS = 1000;
    constexpr const char* EXPOSE_MX_ENV = "BLADED_RL_EXPOSE_MX";
    constexpr const char* LEGACY_UDP_PROTOCOL_VERSION = "bladed_udp_19_si_my_v1";
    constexpr const char* QUEUE_UDP_PROTOCOL_VERSION_V1 = "bladed_udp_19_si_my_queue_v1";
    constexpr const char* QUEUE_UDP_PROTOCOL_VERSION = "bladed_udp_19_si_my_queue_v2";
    constexpr const char* QUEUE_PROBE_COMMAND = "QUEUE_PROBE";
    constexpr const char* QUEUE_ADVANCE_INCOMPLETE_SLOT_COMMAND = "QUEUE_ADVANCE_INCOMPLETE_SLOT";
    constexpr double MANUAL_BATCH_SIM_TIME_RESET_EPSILON_S = 1.0e-6;
    constexpr std::size_t MANUAL_BATCH_CONDITION_COUNT = 23;
    constexpr std::size_t MANUAL_BATCH_DEFAULT_TOTAL_ROUNDS = 4;
    constexpr std::size_t MANUAL_BATCH_ROTATION_STRIDE = 7;
    constexpr const char* MANUAL_BATCH_RUN_ID_ENV = "BLADED_RL_MANUAL_BATCH_RUN_ID";
    constexpr const char* MANUAL_BATCH_PROFILE_ENV = "BLADED_RL_MANUAL_BATCH_PROFILE";
    constexpr const char* MANUAL_BATCH_ROUNDS_ENV = "BLADED_RL_MANUAL_BATCH_ROUNDS";
    constexpr const char* MANUAL_BATCH_PROFILE_FULL = "full_23x4";
    constexpr const char* MANUAL_BATCH_PROFILE_FULL_DYNAMIC = "full";
    constexpr const char* MANUAL_BATCH_PROFILE_SMOKE = "smoke_2x4";
    constexpr const char* MANUAL_BATCH_PROFILE_SMOKE_DYNAMIC = "smoke";
    constexpr const char* MANUAL_BATCH_PROFILE_SMOKE_4WIND = "smoke_4x4";
    constexpr const char* MANUAL_BATCH_STATE_FILE_PREFIX = "bladed_rl_manual_batch_";
    constexpr const char* MANUAL_BATCH_STATE_FILE_SUFFIX = ".state";
    constexpr int MANUAL_BATCH_BASE_CONDITION_ORDER[MANUAL_BATCH_CONDITION_COUNT] = {
        12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6,
        18, 5, 19, 4, 20, 3, 21, 22, 23, 24, 25,
    };
    constexpr std::size_t MANUAL_BATCH_SMOKE_CONDITION_COUNT = 2;
    constexpr std::size_t MANUAL_BATCH_SMOKE_ROTATION_STRIDE = 1;
    constexpr int MANUAL_BATCH_SMOKE_BASE_CONDITION_ORDER[MANUAL_BATCH_SMOKE_CONDITION_COUNT] = { 12, 11 };
    constexpr std::size_t MANUAL_BATCH_SMOKE_4WIND_CONDITION_COUNT = 4;
    constexpr std::size_t MANUAL_BATCH_SMOKE_4WIND_ROTATION_STRIDE = 1;
    constexpr int MANUAL_BATCH_SMOKE_4WIND_BASE_CONDITION_ORDER[
        MANUAL_BATCH_SMOKE_4WIND_CONDITION_COUNT
    ] = { 8, 11, 12, 16 };

    struct ManualBatchSchedule
    {
        const char* profile;
        const int* conditionOrder;
        std::size_t conditionCount;
        std::size_t totalRounds;
        std::size_t totalSlots;
        std::size_t rotationStride;
    };

    struct ManualBatchPersistedState
    {
        std::size_t slotIndex;
        std::uint64_t executionId;
        bool trainCompleted;
    };

    SOCKET udpSocket = INVALID_SOCKET;
    sockaddr_in pythonAddr{};
    bool udpReady = false;
    DWORD udpReceiveTimeoutMs = DEFAULT_UDP_RECEIVE_TIMEOUT_MS;
    std::uint64_t nextFrameId = 1;
    bool udpQueueProtocolEnabled = false;
    bool udpQueueProtocolV2 = false;
    std::uint64_t queueEpoch = 0;
    std::uint64_t conditionInstanceId = 0;
    std::uint64_t conditionId = 0;
    std::string configuredQueuePhase = "AUTO";
    bool udpMxFieldsEnabled = false;
    double postLoadDiscardSimS = 0.0;
    double trainSimS = 0.0;
    bool manualBatchQueueEnabled = false;
    std::size_t manualBatchActiveSlotIndex = 0;
    std::uint64_t manualBatchExecutionId = 0;
    bool manualBatchTrainCompleted = false;
    bool manualBatchHasObservedSimTime = false;
    double manualBatchLastSimTime = 0.0;
    std::string manualBatchStateFilePath;
    bool manualBatchUsesAutomaticRunId = false;
    std::string manualBatchProfileName;
    ManualBatchSchedule manualBatchScheduleStorage{};
    const ManualBatchSchedule* manualBatchSchedule = nullptr;
    std::string udpInitializationError;

    struct QueueFrameMetadata
    {
        const char* phase;
        double phaseElapsedS;
    };

    bool FailUdpInitialization(const char* message)
    {
        udpInitializationError = message;
        return false;
    }

    struct ProcessRecord
    {
        DWORD processId;
        DWORD parentProcessId;
        bool isExplorer;
    };

    std::string GetEnvironmentValueOrDefault(const char* name, const char* defaultValue)
    {
        char* rawValue = nullptr;
        size_t valueLength = 0;
        if (_dupenv_s(&rawValue, &valueLength, name) != 0 || rawValue == nullptr)
            return defaultValue;

        const std::string value = rawValue[0] != '\0' ? rawValue : defaultValue;
        std::free(rawValue);
        return value;
    }

    int GetEnvironmentPortOrDefault(const char* name, int defaultValue)
    {
        const std::string value = GetEnvironmentValueOrDefault(name, "");
        if (value.empty())
            return defaultValue;

        char* end = nullptr;
        const long parsed = std::strtol(value.c_str(), &end, 10);
        return end != value.c_str() && *end == '\0' && parsed > 0 && parsed <= 65535
            ? static_cast<int>(parsed)
            : defaultValue;
    }

    DWORD GetEnvironmentTimeoutOrDefault(const char* name, DWORD defaultValue)
    {
        const int timeout = GetEnvironmentPortOrDefault(name, static_cast<int>(defaultValue));
        return static_cast<DWORD>(timeout);
    }

    std::uint64_t GetEnvironmentUInt64OrDefault(const char* name, std::uint64_t defaultValue)
    {
        const std::string value = GetEnvironmentValueOrDefault(name, "");
        if (value.empty())
            return defaultValue;

        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
        return end != value.c_str() && *end == '\0'
            ? static_cast<std::uint64_t>(parsed)
            : defaultValue;
    }

    double GetEnvironmentNonnegativeDoubleOrDefault(const char* name, double defaultValue)
    {
        const std::string value = GetEnvironmentValueOrDefault(name, "");
        if (value.empty())
            return defaultValue;

        char* end = nullptr;
        const double parsed = std::strtod(value.c_str(), &end);
        return end != value.c_str() && *end == '\0' && std::isfinite(parsed) && parsed >= 0.0
            ? parsed
            : defaultValue;
    }

    std::string ToUpper(std::string value)
    {
        for (char& character : value)
            character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
        return value;
    }

    bool GetEnvironmentBooleanOrDefault(const char* name, bool defaultValue)
    {
        const std::string value = ToUpper(GetEnvironmentValueOrDefault(name, ""));
        if (value.empty())
            return defaultValue;
        if (value == "1" || value == "TRUE" || value == "YES" || value == "ON")
            return true;
        if (value == "0" || value == "FALSE" || value == "NO" || value == "OFF")
            return false;
        return defaultValue;
    }

    bool GetManualBatchRounds(std::size_t& rounds)
    {
        const std::string configured = GetEnvironmentValueOrDefault(MANUAL_BATCH_ROUNDS_ENV, "");
        if (configured.empty())
        {
            rounds = MANUAL_BATCH_DEFAULT_TOTAL_ROUNDS;
            return true;
        }

        errno = 0;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(configured.c_str(), &end, 10);
        if (
            configured.front() == '-' ||
            end == configured.c_str() ||
            *end != '\0' ||
            errno == ERANGE ||
            parsed == 0 ||
            parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
        {
            return false;
        }

        rounds = static_cast<std::size_t>(parsed);
        return true;
    }

    bool ConfigureManualBatchSchedule()
    {
        const std::string profile = GetEnvironmentValueOrDefault(
            MANUAL_BATCH_PROFILE_ENV, MANUAL_BATCH_PROFILE_FULL);
        const bool fullProfile =
            profile == MANUAL_BATCH_PROFILE_FULL || profile == MANUAL_BATCH_PROFILE_FULL_DYNAMIC;
        const bool smokeProfile =
            profile == MANUAL_BATCH_PROFILE_SMOKE || profile == MANUAL_BATCH_PROFILE_SMOKE_DYNAMIC;
        const bool smoke4WindProfile = profile == MANUAL_BATCH_PROFILE_SMOKE_4WIND;
        if (!fullProfile && !smokeProfile && !smoke4WindProfile)
        {
            return FailUdpInitialization(
                "BLADED_RL_MANUAL_BATCH_PROFILE must be full, smoke, full_23x4, smoke_2x4, or smoke_4x4.");
        }

        std::size_t totalRounds = 0;
        if (!GetManualBatchRounds(totalRounds))
        {
            return FailUdpInitialization(
                "BLADED_RL_MANUAL_BATCH_ROUNDS must be a positive integer within the platform size limit.");
        }

        const std::size_t conditionCount = fullProfile
            ? MANUAL_BATCH_CONDITION_COUNT
            : smoke4WindProfile
                ? MANUAL_BATCH_SMOKE_4WIND_CONDITION_COUNT
                : MANUAL_BATCH_SMOKE_CONDITION_COUNT;
        if (totalRounds > std::numeric_limits<std::size_t>::max() / conditionCount)
        {
            return FailUdpInitialization(
                "BLADED_RL_MANUAL_BATCH_ROUNDS is too large for the queue slot count.");
        }

        const char* profilePrefix = fullProfile
            ? "full_23x"
            : smoke4WindProfile ? "smoke_4x" : "smoke_2x";
        manualBatchProfileName = std::string(profilePrefix) +
            std::to_string(totalRounds);
        manualBatchScheduleStorage = {
            manualBatchProfileName.c_str(),
            fullProfile
                ? MANUAL_BATCH_BASE_CONDITION_ORDER
                : smoke4WindProfile
                    ? MANUAL_BATCH_SMOKE_4WIND_BASE_CONDITION_ORDER
                    : MANUAL_BATCH_SMOKE_BASE_CONDITION_ORDER,
            conditionCount,
            totalRounds,
            conditionCount * totalRounds,
            fullProfile
                ? MANUAL_BATCH_ROTATION_STRIDE
                : smoke4WindProfile
                    ? MANUAL_BATCH_SMOKE_4WIND_ROTATION_STRIDE
                    : MANUAL_BATCH_SMOKE_ROTATION_STRIDE,
        };
        manualBatchSchedule = &manualBatchScheduleStorage;
        return true;
    }

    bool IsValidManualBatchRunId(const std::string& runId)
    {
        if (runId.empty() || runId.size() > 80)
            return false;
        for (const unsigned char character : runId)
        {
            if (!std::isalnum(character) && character != '-' && character != '_')
                return false;
        }
        return true;
    }

    DWORD GetDesktopSessionOwnerProcessId()
    {
        const DWORD currentProcessId = GetCurrentProcessId();
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snapshot == INVALID_HANDLE_VALUE)
            return currentProcessId;

        std::vector<ProcessRecord> processes;
        PROCESSENTRY32W entry{};
        entry.dwSize = sizeof(entry);
        if (Process32FirstW(snapshot, &entry))
        {
            do
            {
                processes.push_back({
                    entry.th32ProcessID,
                    entry.th32ParentProcessID,
                    _wcsicmp(entry.szExeFile, L"explorer.exe") == 0,
                });
            } while (Process32NextW(snapshot, &entry));
        }
        CloseHandle(snapshot);

        DWORD sessionOwnerProcessId = currentProcessId;
        for (int depth = 0; depth < 64; ++depth)
        {
            const ProcessRecord* current = nullptr;
            const ProcessRecord* parent = nullptr;
            for (const ProcessRecord& process : processes)
            {
                if (process.processId == sessionOwnerProcessId)
                    current = &process;
            }
            if (current == nullptr || current->parentProcessId == 0 ||
                current->parentProcessId == sessionOwnerProcessId)
            {
                break;
            }
            for (const ProcessRecord& process : processes)
            {
                if (process.processId == current->parentProcessId)
                    parent = &process;
            }
            if (parent == nullptr || parent->isExplorer)
                break;
            sessionOwnerProcessId = parent->processId;
        }
        return sessionOwnerProcessId;
    }

    std::uint64_t GetProcessCreationFileTime(DWORD processId)
    {
        HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, processId);
        if (process == nullptr)
            return 0;

        FILETIME creationTime{};
        FILETIME exitTime{};
        FILETIME kernelTime{};
        FILETIME userTime{};
        const BOOL gotTimes = GetProcessTimes(
            process,
            &creationTime,
            &exitTime,
            &kernelTime,
            &userTime);
        CloseHandle(process);
        if (!gotTimes)
            return 0;

        ULARGE_INTEGER value{};
        value.LowPart = creationTime.dwLowDateTime;
        value.HighPart = creationTime.dwHighDateTime;
        return value.QuadPart;
    }

    std::string GetAutomaticManualBatchRunId()
    {
        const DWORD sessionOwnerProcessId = GetDesktopSessionOwnerProcessId();
        const std::uint64_t creationFileTime = GetProcessCreationFileTime(sessionOwnerProcessId);
        std::string runId = "desktop_" + std::to_string(sessionOwnerProcessId);
        if (creationFileTime != 0)
            runId += "_" + std::to_string(creationFileTime);
        return runId;
    }

    std::string GetManualBatchStateFilePath(const std::string& runId)
    {
        std::string stateDirectory = GetEnvironmentValueOrDefault("LOCALAPPDATA", "");
        if (stateDirectory.empty())
            stateDirectory = GetEnvironmentValueOrDefault("TEMP", "");
        if (stateDirectory.empty())
            return "";

        const char lastCharacter = stateDirectory.back();
        if (lastCharacter != '\\' && lastCharacter != '/')
            stateDirectory += '\\';
        return stateDirectory + MANUAL_BATCH_STATE_FILE_PREFIX + runId + MANUAL_BATCH_STATE_FILE_SUFFIX;
    }

    bool ReadManualBatchPersistedState(ManualBatchPersistedState& state, bool& hasPersistedState)
    {
        hasPersistedState = false;
        if (manualBatchSchedule == nullptr)
            return false;
        errno = 0;
        if (_access(manualBatchStateFilePath.c_str(), 0) != 0)
            return errno == ENOENT;

        std::ifstream input(manualBatchStateFilePath);
        std::string formatVersion;
        unsigned long long persistedSlot = 0;
        unsigned long long persistedExecutionId = 0;
        int persistedTrainCompleted = -1;
        if (
            !input ||
            !(input >> formatVersion >> persistedSlot >> persistedExecutionId >> persistedTrainCompleted) ||
            formatVersion != "v2" ||
            persistedSlot >= manualBatchSchedule->totalSlots ||
            persistedExecutionId == 0 ||
            (persistedTrainCompleted != 0 && persistedTrainCompleted != 1))
        {
            std::printf(
                "Manual Batch Run state is invalid or uses an obsolete format: %s. "
                "Restart Bladed with a new desktop session or BLADED_RL_MANUAL_BATCH_RUN_ID.\n",
                manualBatchStateFilePath.c_str());
            return false;
        }

        state = {
            static_cast<std::size_t>(persistedSlot),
            static_cast<std::uint64_t>(persistedExecutionId),
            persistedTrainCompleted != 0,
        };
        hasPersistedState = true;
        return true;
    }

    bool PersistManualBatchState(const ManualBatchPersistedState& state)
    {
        std::ofstream output(manualBatchStateFilePath, std::ios::trunc);
        if (!output)
            return false;
        output << "v2 " << state.slotIndex << ' ' << state.executionId << ' '
            << (state.trainCompleted ? 1 : 0) << '\n';
        output.flush();
        return output.good();
    }

    void ResetManualBatchQueue()
    {
        manualBatchQueueEnabled = false;
        manualBatchActiveSlotIndex = 0;
        manualBatchExecutionId = 0;
        manualBatchTrainCompleted = false;
        manualBatchHasObservedSimTime = false;
        manualBatchLastSimTime = 0.0;
        manualBatchStateFilePath.clear();
        manualBatchUsesAutomaticRunId = false;
        manualBatchProfileName.clear();
        manualBatchSchedule = nullptr;
    }

    bool ActivateManualBatchQueueSlot(
        std::size_t slotIndex,
        std::uint64_t executionId,
        const char* activationReason)
    {
        if (manualBatchSchedule == nullptr)
            return FailUdpInitialization("Manual Batch Run schedule was not configured.");

        const ManualBatchSchedule& schedule = *manualBatchSchedule;
        if (slotIndex >= schedule.totalSlots)
        {
            std::printf("Manual Batch Run queue is exhausted; restart Bladed before another simulation run.\n");
            return FailUdpInitialization(
                "Manual Batch Run queue is exhausted; use a new BLADED_RL_MANUAL_BATCH_RUN_ID.");
        }
        if (executionId == 0)
            return FailUdpInitialization("Manual Batch Run execution ID must be positive.");

        manualBatchActiveSlotIndex = slotIndex;
        manualBatchExecutionId = executionId;
        manualBatchTrainCompleted = false;
        const std::size_t roundIndex = slotIndex / schedule.conditionCount;
        const std::size_t indexInRound = slotIndex % schedule.conditionCount;
        const std::size_t rotationOffset = (roundIndex * schedule.rotationStride) % schedule.conditionCount;
        const std::size_t conditionIndex = (rotationOffset + indexInRound) % schedule.conditionCount;
        queueEpoch = executionId;
        conditionInstanceId = executionId;
        conditionId = static_cast<std::uint64_t>(schedule.conditionOrder[conditionIndex]);
        nextFrameId = 1;
        if (!PersistManualBatchState({ slotIndex, executionId, false }))
        {
            std::printf("Manual Batch Run cannot write state file: %s\n", manualBatchStateFilePath.c_str());
            return FailUdpInitialization("Manual Batch Run cannot write its state file.");
        }
        std::printf(
            "Bladed manual Batch Run profile=%s slot=%zu/%zu round=%zu epoch=%llu instance=%llu "
            "condition=%llu train=%.3f s activation=%s state=%s\n",
            schedule.profile,
            slotIndex + 1,
            schedule.totalSlots,
            roundIndex + 1,
            static_cast<unsigned long long>(queueEpoch),
            static_cast<unsigned long long>(conditionInstanceId),
            static_cast<unsigned long long>(conditionId),
            trainSimS,
            activationReason,
            manualBatchStateFilePath.c_str());
        return true;
    }

    bool MarkManualBatchTrainCompleted()
    {
        if (!manualBatchQueueEnabled || manualBatchTrainCompleted)
            return true;
        if (!PersistManualBatchState({ manualBatchActiveSlotIndex, manualBatchExecutionId, true }))
        {
            std::printf("Manual Batch Run cannot mark completed TRAIN state: %s\n", manualBatchStateFilePath.c_str());
            return FailUdpInitialization("Manual Batch Run cannot persist completed TRAIN state.");
        }
        manualBatchTrainCompleted = true;
        std::printf(
            "Bladed manual Batch Run completed TRAIN slot=%zu epoch=%llu instance=%llu condition=%llu\n",
            manualBatchActiveSlotIndex + 1,
            static_cast<unsigned long long>(queueEpoch),
            static_cast<unsigned long long>(conditionInstanceId),
            static_cast<unsigned long long>(conditionId));
        return true;
    }

    bool AdvanceManualBatchQueueAfterVerifiedPhysicalSkip()
    {
        if (!manualBatchQueueEnabled || manualBatchTrainCompleted)
            return FailUdpInitialization(
                "Manual Batch Run can only advance an incomplete active slot after a verified physical skip.");
        return ActivateManualBatchQueueSlot(
            manualBatchActiveSlotIndex + 1,
            manualBatchExecutionId + 1,
            "advance_after_verified_physical_skip");
    }

    bool ConfigureManualBatchQueue()
    {
        if (!ConfigureManualBatchSchedule())
            return false;
        std::string runId = GetEnvironmentValueOrDefault(MANUAL_BATCH_RUN_ID_ENV, "");
        manualBatchUsesAutomaticRunId = runId.empty();
        if (manualBatchUsesAutomaticRunId)
        {
            runId = GetAutomaticManualBatchRunId();
            std::printf("Manual Batch Run automatic desktop session=%s\n", runId.c_str());
        }
        if (!IsValidManualBatchRunId(runId))
        {
            std::printf(
                "Manual Batch Run %s must contain 1-80 letters, digits, '-' or '_'.\n",
                manualBatchUsesAutomaticRunId ? "automatic session ID" : MANUAL_BATCH_RUN_ID_ENV);
            return FailUdpInitialization(
                manualBatchUsesAutomaticRunId
                    ? "Manual Batch Run could not create an automatic desktop session ID."
                    : "BLADED_RL_MANUAL_BATCH_RUN_ID must contain letters, digits, '-' or '_'.");
        }
        manualBatchStateFilePath = GetManualBatchStateFilePath(
            runId + "_" + manualBatchSchedule->profile);
        if (manualBatchStateFilePath.empty())
        {
            std::printf("Manual Batch Run cannot determine a state-file directory.\n");
            return FailUdpInitialization("Manual Batch Run cannot resolve LOCALAPPDATA or TEMP for its state file.");
        }

        ManualBatchPersistedState persistedState{};
        bool hasPersistedState = false;
        if (!ReadManualBatchPersistedState(persistedState, hasPersistedState))
        {
            std::printf("Manual Batch Run cannot read state file: %s\n", manualBatchStateFilePath.c_str());
            return FailUdpInitialization("Manual Batch Run cannot read its state file.");
        }
        if (
            hasPersistedState &&
            persistedState.trainCompleted &&
            persistedState.slotIndex + 1 >= manualBatchSchedule->totalSlots)
        {
            std::printf(
                "Manual Batch Run state is exhausted; %s.\n",
                manualBatchUsesAutomaticRunId
                    ? "close Bladed and start it again from the desktop icon"
                    : "use a new BLADED_RL_MANUAL_BATCH_RUN_ID");
            return FailUdpInitialization(
                manualBatchUsesAutomaticRunId
                    ? "Manual Batch Run state is exhausted; close Bladed and start it again from the desktop icon."
                    : "Manual Batch Run state is exhausted; use a new BLADED_RL_MANUAL_BATCH_RUN_ID.");
        }

        manualBatchQueueEnabled = true;
        postLoadDiscardSimS = GetEnvironmentNonnegativeDoubleOrDefault(
            "BLADED_RL_POST_LOAD_DISCARD_SIM_S", 0.0);
        trainSimS = GetEnvironmentNonnegativeDoubleOrDefault("BLADED_RL_TRAIN_SIM_S", 600.0);
        if (trainSimS <= 0.0)
            return FailUdpInitialization("BLADED_RL_TRAIN_SIM_S must be greater than zero.");
        configuredQueuePhase = "AUTO";
        manualBatchActiveSlotIndex = 0;
        manualBatchExecutionId = 0;
        manualBatchTrainCompleted = false;
        manualBatchHasObservedSimTime = false;
        manualBatchLastSimTime = 0.0;
        const std::size_t slotIndex = hasPersistedState
            ? persistedState.slotIndex + (persistedState.trainCompleted ? 1 : 0)
            : 0;
        const std::uint64_t executionId = hasPersistedState ? persistedState.executionId + 1 : 1;
        return ActivateManualBatchQueueSlot(
            slotIndex,
            executionId,
            hasPersistedState
                ? (persistedState.trainCompleted ? "advance_after_complete" : "retry_after_incomplete")
                : "initial");
    }

    bool AdvanceManualBatchQueueOnSimulationReset(double simTime)
    {
        if (!manualBatchQueueEnabled)
            return true;
        if (!std::isfinite(simTime) || simTime < 0.0)
            return false;
        if (
            manualBatchHasObservedSimTime &&
            simTime + MANUAL_BATCH_SIM_TIME_RESET_EPSILON_S < manualBatchLastSimTime)
        {
            const bool advanceSlot = manualBatchTrainCompleted;
            if (!ActivateManualBatchQueueSlot(
                    manualBatchActiveSlotIndex + (advanceSlot ? 1 : 0),
                    manualBatchExecutionId + 1,
                    advanceSlot ? "advance_on_reset" : "retry_on_reset"))
            {
                return false;
            }
        }
        manualBatchLastSimTime = simTime;
        manualBatchHasObservedSimTime = true;
        return true;
    }

    bool ConfigureQueueProtocol()
    {
        ResetManualBatchQueue();
        udpQueueProtocolV2 = false;
        const std::string protocol = GetEnvironmentValueOrDefault(
            "BLADED_RL_UDP_PROTOCOL", QUEUE_UDP_PROTOCOL_VERSION);
        if (protocol == LEGACY_UDP_PROTOCOL_VERSION)
        {
            udpQueueProtocolEnabled = false;
            return true;
        }
        if (protocol != QUEUE_UDP_PROTOCOL_VERSION_V1 && protocol != QUEUE_UDP_PROTOCOL_VERSION)
            return FailUdpInitialization(
                "BLADED_RL_UDP_PROTOCOL must be bladed_udp_19_si_my_v1, "
                "bladed_udp_19_si_my_queue_v1, or bladed_udp_19_si_my_queue_v2.");

        udpQueueProtocolEnabled = true;
        udpQueueProtocolV2 = protocol == QUEUE_UDP_PROTOCOL_VERSION;
        const std::string queueEpochValue = GetEnvironmentValueOrDefault("BLADED_RL_QUEUE_EPOCH", "");
        const std::string conditionInstanceValue = GetEnvironmentValueOrDefault(
            "BLADED_RL_CONDITION_INSTANCE_ID", "");
        const std::string conditionValue = GetEnvironmentValueOrDefault("BLADED_RL_CONDITION_ID", "");
        const bool hasAnyQueueIdentity =
            !queueEpochValue.empty() || !conditionInstanceValue.empty() || !conditionValue.empty();
        const bool hasAllQueueIdentity =
            !queueEpochValue.empty() && !conditionInstanceValue.empty() && !conditionValue.empty();
        if (!hasAnyQueueIdentity)
            return ConfigureManualBatchQueue();
        if (!hasAllQueueIdentity)
            return FailUdpInitialization(
                "Set all queue identity variables or clear all three for manual Batch Run mode.");

        queueEpoch = GetEnvironmentUInt64OrDefault("BLADED_RL_QUEUE_EPOCH", 0);
        conditionInstanceId = GetEnvironmentUInt64OrDefault("BLADED_RL_CONDITION_INSTANCE_ID", 0);
        conditionId = GetEnvironmentUInt64OrDefault("BLADED_RL_CONDITION_ID", 0);
        configuredQueuePhase = ToUpper(GetEnvironmentValueOrDefault("BLADED_RL_QUEUE_PHASE", "AUTO"));
        if (
            configuredQueuePhase != "AUTO" &&
            configuredQueuePhase != "SETTLE" &&
            configuredQueuePhase != "TRAIN" &&
            configuredQueuePhase != "END")
        {
            return FailUdpInitialization("BLADED_RL_QUEUE_PHASE must be AUTO, SETTLE, TRAIN, or END.");
        }
        postLoadDiscardSimS = GetEnvironmentNonnegativeDoubleOrDefault(
            "BLADED_RL_POST_LOAD_DISCARD_SIM_S", 0.0);
        trainSimS = GetEnvironmentNonnegativeDoubleOrDefault("BLADED_RL_TRAIN_SIM_S", 0.0);
        return true;
    }

    QueueFrameMetadata GetQueueFrameMetadata(double simTime, double outputStartTime)
    {
        const double activeElapsedS = simTime > outputStartTime ? simTime - outputStartTime : 0.0;
        if (configuredQueuePhase == "SETTLE")
            return { "SETTLE", activeElapsedS };
        if (configuredQueuePhase == "TRAIN")
            return { "TRAIN", activeElapsedS };
        if (configuredQueuePhase == "END")
            return { "END", activeElapsedS };
        if (activeElapsedS < postLoadDiscardSimS)
            return { "SETTLE", activeElapsedS };

        const double trainElapsedS = activeElapsedS - postLoadDiscardSimS;
        if (trainSimS > 0.0 && trainElapsedS >= trainSimS)
            return { "END", trainElapsedS - trainSimS };
        return { "TRAIN", trainElapsedS };
    }

    void CleanupUDP()
    {
        if (udpSocket != INVALID_SOCKET)
        {
            closesocket(udpSocket);
            udpSocket = INVALID_SOCKET;
        }
        WSACleanup();
        udpReady = false;
    }

    bool InitUDP()
    {
        udpInitializationError.clear();
        udpMxFieldsEnabled = GetEnvironmentBooleanOrDefault(EXPOSE_MX_ENV, false);
        const std::string serverIp = GetEnvironmentValueOrDefault(
            "BLADED_RL_HOST", DEFAULT_UDP_SERVER_IP);
        const int serverPort = GetEnvironmentPortOrDefault("BLADED_RL_PORT", DEFAULT_UDP_SERVER_PORT);
        const DWORD receiveTimeoutMs = GetEnvironmentTimeoutOrDefault(
            "BLADED_RL_TIMEOUT_MS", DEFAULT_UDP_RECEIVE_TIMEOUT_MS);
        udpReceiveTimeoutMs = receiveTimeoutMs;
        if (!ConfigureQueueProtocol())
        {
            if (udpInitializationError.empty())
                FailUdpInitialization("UDP queue protocol configuration failed.");
            return false;
        }

        WSADATA wsaData{};
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
            return FailUdpInitialization("Winsock initialization failed.");

        udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (udpSocket == INVALID_SOCKET)
        {
            WSACleanup();
            return FailUdpInitialization("UDP socket creation failed.");
        }

        pythonAddr.sin_family = AF_INET;
        pythonAddr.sin_port = htons(static_cast<u_short>(serverPort));
        if (inet_pton(AF_INET, serverIp.c_str(), &pythonAddr.sin_addr) != 1)
        {
            CleanupUDP();
            return FailUdpInitialization("BLADED_RL_HOST is not a valid IPv4 address.");
        }

        if (connect(udpSocket, reinterpret_cast<sockaddr*>(&pythonAddr), sizeof(pythonAddr)) == SOCKET_ERROR)
        {
            CleanupUDP();
            return FailUdpInitialization("UDP connect to BLADED_RL_HOST:BLADED_RL_PORT failed.");
        }

        setsockopt(
            udpSocket,
            SOL_SOCKET,
            SO_RCVTIMEO,
            reinterpret_cast<const char*>(&receiveTimeoutMs),
            sizeof(receiveTimeoutMs));

        std::printf(
            "Bladed UDP target=%s:%d timeout=%lu ms protocol=%s payload_fields=%d expose_mx=%s\n",
            serverIp.c_str(),
            serverPort,
            static_cast<unsigned long>(receiveTimeoutMs),
            udpQueueProtocolEnabled
                ? (udpQueueProtocolV2 ? QUEUE_UDP_PROTOCOL_VERSION : QUEUE_UDP_PROTOCOL_VERSION_V1)
                : LEGACY_UDP_PROTOCOL_VERSION,
            udpMxFieldsEnabled ? UDP_FIELD_COUNT : LEGACY_UDP_FIELD_COUNT,
            udpMxFieldsEnabled ? "true" : "false");
        if (udpQueueProtocolEnabled)
        {
            std::printf(
                "Bladed queue epoch=%llu condition_instance=%llu condition=%llu phase=%s post_load_discard=%.3f s train=%.3f s\n",
                static_cast<unsigned long long>(queueEpoch),
                static_cast<unsigned long long>(conditionInstanceId),
                static_cast<unsigned long long>(conditionId),
                configuredQueuePhase.c_str(),
                postLoadDiscardSimS,
                trainSimS);
            if (manualBatchQueueEnabled)
            {
                std::printf(
                    "Bladed manual Batch Run built-in schedule rounds=%zu slots=%zu profile=%s active_slot=%zu\n",
                    manualBatchSchedule->totalRounds,
                    manualBatchSchedule->totalSlots,
                    manualBatchSchedule->profile,
                    manualBatchActiveSlotIndex + 1);
            }
        }

        udpReady = true;
        return true;
    }

    void SendToPython(const char* data)
    {
        if (!udpReady || data == nullptr) return;
        send(udpSocket, data, static_cast<int>(strlen(data)), 0);
    }

    std::string RecvFromPython()
    {
        if (!udpReady) return "";
        char buf[4096] = { 0 };
        int n = recv(udpSocket, buf, static_cast<int>(sizeof(buf) - 1), 0);
        if (n <= 0)
            return "";
        buf[n] = '\0';
        return std::string(buf);
    }

    enum class UdpReplyKind
    {
        Action,
        Probe,
        AdvanceIncompleteSlot,
    };

    bool TryParseMatchingReply(
        const std::string& reply,
        std::uint64_t expectedQueueEpoch,
        std::uint64_t expectedConditionInstanceId,
        std::uint64_t expectedFrameId,
        double& torque,
        double& pitch,
        UdpReplyKind& replyKind)
    {
        if (udpQueueProtocolEnabled)
        {
            unsigned long long replyQueueEpoch = 0;
            unsigned long long replyConditionInstanceId = 0;
            unsigned long long replyFrameId = 0;
            double parsedTorque = 0.0;
            double parsedPitch = 0.0;
            const int parsed = sscanf_s(
                reply.c_str(),
                "%llu,%llu,%llu,%lf,%lf",
                &replyQueueEpoch,
                &replyConditionInstanceId,
                &replyFrameId,
                &parsedTorque,
                &parsedPitch);
            if (
                parsed != 5 ||
                replyQueueEpoch != static_cast<unsigned long long>(expectedQueueEpoch) ||
                replyConditionInstanceId != static_cast<unsigned long long>(expectedConditionInstanceId) ||
                replyFrameId != static_cast<unsigned long long>(expectedFrameId) ||
                !std::isfinite(parsedTorque) ||
                !std::isfinite(parsedPitch))
            {
                if (!udpQueueProtocolV2)
                    return false;

                char command[64] = { 0 };
                const int parsedCommand = sscanf_s(
                    reply.c_str(),
                    "%llu,%llu,%llu,%63s",
                    &replyQueueEpoch,
                    &replyConditionInstanceId,
                    &replyFrameId,
                    command,
                    static_cast<unsigned>(sizeof(command)));
                if (
                    parsedCommand != 4 ||
                    replyQueueEpoch != static_cast<unsigned long long>(expectedQueueEpoch) ||
                    replyConditionInstanceId != static_cast<unsigned long long>(expectedConditionInstanceId) ||
                    replyFrameId != static_cast<unsigned long long>(expectedFrameId))
                {
                    return false;
                }
                if (std::strcmp(command, QUEUE_PROBE_COMMAND) == 0)
                {
                    replyKind = UdpReplyKind::Probe;
                    return true;
                }
                if (std::strcmp(command, QUEUE_ADVANCE_INCOMPLETE_SLOT_COMMAND) == 0)
                {
                    replyKind = UdpReplyKind::AdvanceIncompleteSlot;
                    return true;
                }
                return false;
            }

            torque = parsedTorque;
            pitch = parsedPitch;
            replyKind = UdpReplyKind::Action;
            return true;
        }

        unsigned long long replyFrameId = 0;
        double parsedTorque = 0.0;
        double parsedPitch = 0.0;
        const int parsed = sscanf_s(
            reply.c_str(),
            "%llu,%lf,%lf",
            &replyFrameId,
            &parsedTorque,
            &parsedPitch);
        if (
            parsed != 3 ||
            replyFrameId != static_cast<unsigned long long>(expectedFrameId) ||
            !std::isfinite(parsedTorque) ||
            !std::isfinite(parsedPitch))
        {
            return false;
        }

        torque = parsedTorque;
        pitch = parsedPitch;
        replyKind = UdpReplyKind::Action;
        return true;
    }

    bool ReceiveMatchingReply(
        std::uint64_t expectedQueueEpoch,
        std::uint64_t expectedConditionInstanceId,
        std::uint64_t expectedFrameId,
        double& torque,
        double& pitch,
        UdpReplyKind& replyKind)
    {
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(udpReceiveTimeoutMs);
        while (std::chrono::steady_clock::now() < deadline)
        {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            const DWORD remainingTimeoutMs = static_cast<DWORD>(remaining > 1 ? remaining : 1);
            setsockopt(
                udpSocket,
                SOL_SOCKET,
                SO_RCVTIMEO,
                reinterpret_cast<const char*>(&remainingTimeoutMs),
                sizeof(remainingTimeoutMs));

            const std::string reply = RecvFromPython();
            if (reply.empty())
            {
                return false;
            }
            if (TryParseMatchingReply(
                reply,
                expectedQueueEpoch,
                expectedConditionInstanceId,
                expectedFrameId,
                torque,
                pitch,
                replyKind))
            {
                return true;
            }

            std::printf(
                "Ignored stale or malformed UDP action while waiting for epoch=%llu instance=%llu frame=%llu\n",
                static_cast<unsigned long long>(expectedQueueEpoch),
                static_cast<unsigned long long>(expectedConditionInstanceId),
                static_cast<unsigned long long>(expectedFrameId));
        }
        return false;
    }

    double GetBladeRootOutOfPlaneMoment(const turbine tid, int bladeIndex)
    {
        return GetMeasuredBladeOutOfPlaneBendingMoment(tid, bladeIndex);
    }
    double GetBladeRootInPlaneMoment(const turbine tid, int bladeIndex)
    {
        return GetMeasuredBladeInPlaneBendingMoment(tid, bladeIndex);
    }

    bool ApplyCollectivePitchDemand(const turbine tid, double pitch)
    {
        const int pitchControl = GetPitchControl(tid);
        if (pitchControl != GH_DISCON_PITCH_CONTROL_COLLECTIVE)
        {
            std::printf(
                "Bladed pitch control mode is %d; collective pitch is required.\n",
                pitchControl);
            return false;
        }

        const int result = SetDemandedCollectivePitchAngle(tid, pitch);
        if (result != GH_DISCON_SUCCESS)
        {
            std::printf(
                "SetDemandedCollectivePitchAngle failed: result=%d pitch=%.9f rad\n",
                result,
                pitch);
            return false;
        }

        const double demandedPitch = GetDemandedCollectivePitchAngle(tid);
        if (!std::isfinite(demandedPitch) || std::abs(demandedPitch - pitch) > 1.0e-7)
        {
            std::printf(
                "Collective pitch demand readback mismatch: requested=%.9f rad readback=%.9f rad\n",
                pitch,
                demandedPitch);
            return false;
        }
        return true;
    }

    void SetField(std::vector<std::string>& fields, int index, double value)
    {
        char buf[64] = { 0 };
        sprintf_s(buf, sizeof(buf), "%.6f", value);
        fields[static_cast<size_t>(index)] = buf;
    }

    std::string BuildCsvLine(const std::vector<std::string>& fields, std::size_t fieldCount)
    {
        std::string line;
        line.reserve(8192);
        for (size_t i = 0; i < fieldCount; ++i)
        {
            if (i > 0)
                line += ',';
            line += fields[i];
        }
        return line;
    }
}

extern "C"
{
    int __declspec(dllexport) __cdecl CONTROLLER(const turbine turbine_id)
    {
        // GetSimulationCurrentTime：仿真步长时序 0 / 0.02 / 0.04 ...
        const double simTime = GetSimulationCurrentTime(turbine_id);
        if (
            udpQueueProtocolEnabled &&
            manualBatchQueueEnabled &&
            !AdvanceManualBatchQueueOnSimulationReset(simTime))
        {
            ReportErrorMessage(turbine_id, "Manual Batch Run queue cannot activate another condition.");
            return GH_DISCON_ERROR;
        }
        const double outputStartTime = GetSettlingTime(turbine_id);
        // 修改1：Time = 输出相对时间 simTime - outputStartTime
        const double Time = simTime - outputStartTime;

        const double I_RotorAzimuth = GetMeasuredRotorAzimuthAngle(turbine_id);
        const double I_RotorSpeed = GetMeasuredRotorSpeed(turbine_id);
        const double I_AverageWindSpeed = GetNominalHubFlowSpeed(turbine_id);
        const double I_YawError = GetMeasuredYawError(turbine_id);
        const double I_ElectricalTorque = GetMeasuredGeneratorTorque(turbine_id);
        const double I_GeneratorSpeed = GetMeasuredGeneratorSpeed(turbine_id);
        const double I_RealPower = GetMeasuredElectricalPowerOutput(turbine_id);
        const double I_CollectivePitchPosition = GetCollectivePitchAngle(turbine_id);
        const double I_PitchPosition1 = I_CollectivePitchPosition;
        const double I_PitchPosition2 = I_CollectivePitchPosition;
        const double I_PitchPosition3 = I_CollectivePitchPosition;
        const double I_NacelleForeAftAcceleration = GetMeasuredTowerTopForeAftAcceleration(turbine_id);
        const double I_NacelleSideSideAcceleration = GetMeasuredTowerTopSideSideAcceleration(turbine_id);

        const double blade1My = GetBladeRootOutOfPlaneMoment(turbine_id, 0);
        const double blade2My = GetBladeRootOutOfPlaneMoment(turbine_id, 1);
        const double blade3My = GetBladeRootOutOfPlaneMoment(turbine_id, 2);

        const double blade1Mx = GetBladeRootInPlaneMoment(turbine_id, 0);
        const double blade2Mx = GetBladeRootInPlaneMoment(turbine_id, 1);
        const double blade3Mx = GetBladeRootInPlaneMoment(turbine_id, 2);

        double new_torque = I_ElectricalTorque;
        double new_pitch = I_CollectivePitchPosition;

        // 修改2：未到输出10s直接返回，不组装报文、不发送UDP，CSV无负数
        if (simTime < outputStartTime)
        {
            SetDemandedGeneratorTorque(turbine_id, new_torque);
            if (!ApplyCollectivePitchDemand(turbine_id, new_pitch))
                return GH_DISCON_ERROR;
            return GH_DISCON_SUCCESS;
        }

        if (!udpReady)
        {
            SetDemandedGeneratorTorque(turbine_id, new_torque);
            if (!ApplyCollectivePitchDemand(turbine_id, new_pitch))
                return GH_DISCON_ERROR;
            return GH_DISCON_SUCCESS;
        }

        const auto tStart = std::chrono::steady_clock::now();
        std::vector<std::string> fields(static_cast<size_t>(UDP_FIELD_COUNT));

        SetField(fields, kTime, Time);
        SetField(fields, kRotorAzimuth, I_RotorAzimuth);
        SetField(fields, kRotorSpeed, I_RotorSpeed);
        SetField(fields, kYawError, I_YawError);
        SetField(fields, kElectricalTorque, I_ElectricalTorque);
        SetField(fields, kTorqueDemand, I_ElectricalTorque);
        SetField(fields, kGeneratorSpeed, I_GeneratorSpeed);
        SetField(fields, kPitchPosition1, I_PitchPosition1);
        SetField(fields, kPitchDemand1, I_PitchPosition1);
        SetField(fields, kPitchPosition2, I_PitchPosition2);
        SetField(fields, kPitchPosition3, I_PitchPosition3);
        SetField(fields, kNacelleForeAftAcceleration, I_NacelleForeAftAcceleration);
        SetField(fields, kNacelleSideSideAcceleration, I_NacelleSideSideAcceleration);
        SetField(fields, kAverageWindSpeed, I_AverageWindSpeed);
        SetField(fields, kElectricalPower, I_RealPower);
        SetField(fields, kRealPower, I_RealPower);
        SetField(fields, kBladeRoot1My, blade1My);
        SetField(fields, kBladeRoot2My, blade2My);
        SetField(fields, kBladeRoot3My, blade3My);
        SetField(fields, kBladeRoot1Mx, blade1Mx);
        SetField(fields, kBladeRoot2Mx, blade2Mx);
        SetField(fields, kBladeRoot3Mx, blade3Mx);

        const std::uint64_t frameId = nextFrameId++;
        const std::size_t wireFieldCount = udpMxFieldsEnabled
            ? static_cast<std::size_t>(UDP_FIELD_COUNT)
            : static_cast<std::size_t>(LEGACY_UDP_FIELD_COUNT);
        const std::string csvLine = BuildCsvLine(fields, wireFieldCount);
        const QueueFrameMetadata queueMetadata = GetQueueFrameMetadata(simTime, outputStartTime);
        if (std::strcmp(queueMetadata.phase, "END") == 0 && !MarkManualBatchTrainCompleted())
        {
            ReportErrorMessage(
                turbine_id,
                udpInitializationError.empty()
                    ? "Manual Batch Run cannot persist completed TRAIN state."
                    : udpInitializationError.c_str());
            return GH_DISCON_ERROR;
        }
        char framePrefix[256] = { 0 };
        if (udpQueueProtocolEnabled)
        {
            if (udpQueueProtocolV2)
            {
                sprintf_s(
                    framePrefix,
                    sizeof(framePrefix),
                    "%s,%llu,%llu,%llu,%s,%.6f,%llu,%.6f,",
                    QUEUE_UDP_PROTOCOL_VERSION,
                    static_cast<unsigned long long>(queueEpoch),
                    static_cast<unsigned long long>(conditionInstanceId),
                    static_cast<unsigned long long>(conditionId),
                    queueMetadata.phase,
                    queueMetadata.phaseElapsedS,
                    static_cast<unsigned long long>(frameId),
                    simTime);
            }
            else
            {
                sprintf_s(
                    framePrefix,
                    sizeof(framePrefix),
                    "%llu,%llu,%llu,%s,%.6f,%llu,%.6f,",
                    static_cast<unsigned long long>(queueEpoch),
                    static_cast<unsigned long long>(conditionInstanceId),
                    static_cast<unsigned long long>(conditionId),
                    queueMetadata.phase,
                    queueMetadata.phaseElapsedS,
                    static_cast<unsigned long long>(frameId),
                    simTime);
            }
        }
        else
        {
            sprintf_s(
                framePrefix,
                sizeof(framePrefix),
                "%llu,%.6f,",
                static_cast<unsigned long long>(frameId),
                simTime);
        }
        const std::string udpFrame = std::string(framePrefix) + csvLine;
        std::printf(
            "Bladed输出: Epoch=%llu Instance=%llu Condition=%llu Phase=%s Frame=%llu SimTime=%.6f s, Torque=%.3f Nm, Pitch=%.3f rad, AvgWind=%.3f m/s\n",
            static_cast<unsigned long long>(queueEpoch),
            static_cast<unsigned long long>(conditionInstanceId),
            static_cast<unsigned long long>(conditionId),
            udpQueueProtocolEnabled ? queueMetadata.phase : "LEGACY",
            static_cast<unsigned long long>(frameId),
            simTime,
            I_ElectricalTorque,
            I_PitchPosition1,
            I_AverageWindSpeed);
        SendToPython(udpFrame.c_str());

        UdpReplyKind replyKind = UdpReplyKind::Action;
        if (!ReceiveMatchingReply(queueEpoch, conditionInstanceId, frameId, new_torque, new_pitch, replyKind))
        {
            new_torque = I_ElectricalTorque;
            new_pitch = I_CollectivePitchPosition;
        }
        else if (replyKind == UdpReplyKind::Probe)
        {
            new_torque = I_ElectricalTorque;
            new_pitch = I_CollectivePitchPosition;
        }
        else if (replyKind == UdpReplyKind::AdvanceIncompleteSlot)
        {
            if (!AdvanceManualBatchQueueAfterVerifiedPhysicalSkip())
            {
                ReportErrorMessage(
                    turbine_id,
                    udpInitializationError.empty()
                        ? "Manual Batch Run cannot advance after a verified physical skip."
                        : udpInitializationError.c_str());
                return GH_DISCON_ERROR;
            }
            new_torque = I_ElectricalTorque;
            new_pitch = I_CollectivePitchPosition;
        }

        std::printf(
            "Bladed接收: Epoch=%llu Instance=%llu Frame=%llu SimTime=%.6f s, Torque=%.3f Nm, Pitch=%.3f rad\n",
            static_cast<unsigned long long>(queueEpoch),
            static_cast<unsigned long long>(conditionInstanceId),
            static_cast<unsigned long long>(frameId),
            simTime,
            new_torque,
            new_pitch);

        const auto tEnd = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
        if (tEnd >= static_cast<double>(udpReceiveTimeoutMs) / 1000.0)
        {
            std::printf(
                "处理时间超过%lu ms，保持当前实测扭矩和桨距角\n",
                static_cast<unsigned long>(udpReceiveTimeoutMs));
            SetDemandedGeneratorTorque(turbine_id, I_ElectricalTorque);
            if (!ApplyCollectivePitchDemand(turbine_id, I_CollectivePitchPosition))
                return GH_DISCON_ERROR;
            return GH_DISCON_SUCCESS;
        }

        SetDemandedGeneratorTorque(turbine_id, new_torque);
        if (!ApplyCollectivePitchDemand(turbine_id, new_pitch))
            return GH_DISCON_ERROR;

        return GH_DISCON_SUCCESS;
    }

    int __declspec(dllexport) __cdecl CONTROLLER_INIT(const turbine turbine_id)
    {
        try
        {
            static bool udpInited = false;
            if (!udpInited)
            {
                if (!InitUDP())
                {
                    ReportErrorMessage(
                        turbine_id,
                        udpInitializationError.empty()
                            ? "UDP initialization failed."
                            : udpInitializationError.c_str());
                    return GH_DISCON_ERROR;
                }
                std::atexit(CleanupUDP);
                udpInited = true;
            }

            return CONTROLLER(turbine_id);
        }
        catch (const std::exception& e)
        {
            ReportErrorMessage(turbine_id, e.what());
            return GH_DISCON_ERROR;
        }
    }
}