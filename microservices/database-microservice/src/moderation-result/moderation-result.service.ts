// moderation-result.service.ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { ClassificationResult } from './entities/classification-results.entity';
import { LabelProbability } from './entities/label-probabilities.entity';

@Injectable()
export class ModerationResultService {
  constructor(
    @InjectRepository(ClassificationResult)
    private moderationResultRepository: Repository<ClassificationResult>,
    @InjectRepository(LabelProbability)
    private labelProbabilityRepository: Repository<LabelProbability>,
  ) {}

  async createModerationResult(data: any): Promise<ClassificationResult> {
    const {
      originalPayload: { taskId, userId, classifier, location },
      clipPrediction,
      clipLabelProbabilities,
    } = data;

    const moderationResult = new ClassificationResult();
    moderationResult.taskId = taskId;
    moderationResult.userId = userId;
    moderationResult.classifier = classifier;
    moderationResult.location = location;
    moderationResult.prediction = clipPrediction;

    const savedResult = await this.moderationResultRepository.save(
      moderationResult,
    );

    const labelProbabilities = Object.entries(clipLabelProbabilities).map(
      ([label, probability]) => {
        const labelProbability = new LabelProbability();
        labelProbability.classificationResultId = savedResult.id;
        labelProbability.label = label;
        labelProbability.probability = probability as any;
        return labelProbability;
      },
    );

    await this.labelProbabilityRepository.save(labelProbabilities);

    return savedResult;
  }
}
